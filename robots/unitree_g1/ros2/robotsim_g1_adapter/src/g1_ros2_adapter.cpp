#include "robotsim_g1_adapter/g1_joint_map.hpp"
#include "robotsim_g1_adapter/sim_time.hpp"
#include "robotsim_core/latest_command_buffer.hpp"
#include "robotsim_core/state_timeline.hpp"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <robotsim_interfaces/msg/robot_command.hpp>
#include <robotsim_interfaces/msg/robot_state.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
using RobotCommand = robotsim_interfaces::msg::RobotCommand;
using RobotState = robotsim_interfaces::msg::RobotState;
using SteadyClock = std::chrono::steady_clock;

constexpr int kDdsDomain = 73;
constexpr const char* kDdsInterface = "lo";
constexpr const char* kModelIdentity =
    "unitree_g1:unitree_mujoco@1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d:scene_29dof";
constexpr const char* kStateTopic = "/simulation/ground_truth/g1/state";
constexpr const char* kCommandTopic = "/robot/g1/command";

bool crcValid(const LowState& message) {
  auto copy = message;
  return copy.crc() == crc32_core(reinterpret_cast<uint32_t*>(&copy),
                                  (sizeof(LowState) >> 2) - 1);
}

class G1Ros2Adapter final : public rclcpp::Node {
 public:
  G1Ros2Adapter()
      : Node("robotsim_g1_adapter"),
        tf_static_(std::make_unique<tf2_ros::StaticTransformBroadcaster>(this)) {
    const auto state_origin = declare_parameter<std::string>("state_origin", "");
    const auto timeout_ms = declare_parameter<int>("command_timeout_ms", 250);
    const auto max_effort_nm = declare_parameter<double>("max_effort_nm", 0.2);
    if (state_origin != "simulator_ground_truth")
      throw std::invalid_argument("state_origin must explicitly be simulator_ground_truth");
    if (timeout_ms <= 0 || !std::isfinite(max_effort_nm) || max_effort_nm <= 0.0)
      throw std::invalid_argument("command timeout and effort limit must be positive");
    command_buffer_ = std::make_unique<robotsim_core::LatestCommandBuffer>(
        kModelIdentity, robotsim_g1_adapter::kG1JointNames.size(), max_effort_nm,
        std::chrono::milliseconds(timeout_ms));

    const auto state_qos = rclcpp::QoS(rclcpp::KeepLast(1))
                               .best_effort()
                               .durability_volatile();
    const auto command_qos = rclcpp::QoS(rclcpp::KeepLast(1))
                                 .reliable()
                                 .durability_volatile();
    state_publisher_ = create_publisher<RobotState>(kStateTopic, state_qos);
    command_subscription_ = create_subscription<RobotCommand>(
        kCommandTopic, command_qos,
        [this](RobotCommand::ConstSharedPtr message) { onCommand(*message); });
    clock_publisher_ = create_publisher<rosgraph_msgs::msg::Clock>("/clock", rclcpp::ClockQoS());

    unitree::robot::ChannelFactory::Instance()->Init(kDdsDomain, kDdsInterface);
    low_state_subscriber_ = std::make_unique<unitree::robot::ChannelSubscriber<LowState>>(
        "rt/lowstate");
    low_state_subscriber_->InitChannel(
        [this](const void* raw) { onLowState(raw); }, 1);
    low_command_publisher_ = std::make_unique<unitree::robot::ChannelPublisher<LowCmd>>(
        "rt/lowcmd");
    low_command_publisher_->InitChannel();

    publishStaticFrames();
    state_timer_ = create_wall_timer(std::chrono::milliseconds(1), [this] { publishState(); });
    command_timer_ = create_wall_timer(std::chrono::milliseconds(10), [this] { publishCommand(); });
    RCLCPP_INFO(get_logger(), "G1 SDK2 adapter ready: domain=%d interface=%s",
                kDdsDomain, kDdsInterface);
  }

  ~G1Ros2Adapter() override {
    command_timer_.reset();
    state_timer_.reset();
    low_state_subscriber_.reset();
    low_command_publisher_.reset();
  }

 private:
  void publishStaticFrames() {
    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = "base_link";
    transform.child_frame_id = "imu_link";
    transform.transform.rotation.w = 1.0;
    tf_static_->sendTransform(transform);
  }

  void onLowState(const void* raw) {
    if (raw == nullptr) return;
    LowState copy = *static_cast<const LowState*>(raw);
    if (!crcValid(copy)) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      latest_low_state_ = std::move(copy);
      have_low_state_ = true;
      ++state_revision_;
    }
  }

  void onCommand(const RobotCommand& message) {
    if (!timeline_initialized_) {
      RCLCPP_WARN(get_logger(), "rejecting command before first simulator state");
      return;
    }
    try {
      (void)robotsim_g1_adapter::toSdkOrder(message.joint_names, message.effort_nm);
    } catch (const std::exception& error) {
      RCLCPP_WARN(get_logger(), "rejecting command joint map: %s", error.what());
      return;
    }
    std::string reason;
    if (!command_buffer_->accept(message, timeline_.generation(), SteadyClock::now(), &reason))
      RCLCPP_WARN(get_logger(), "rejecting command: %s", reason.c_str());
  }

  void publishState() {
    LowState low_state;
    uint64_t revision;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!have_low_state_ || state_revision_ == published_state_revision_) return;
      low_state = latest_low_state_;
      revision = state_revision_;
    }
    published_state_revision_ = revision;

    const auto stamp = robotsim_g1_adapter::tickMillisecondsToRosTime(low_state.tick());
    const int64_t sim_time_ns = static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
    const auto state_stamp = timeline_.next(sim_time_ns);
    const bool first_state = !timeline_initialized_;
    if (!timeline_initialized_ || state_stamp.generation_changed) {
      command_buffer_->resetGeneration(state_stamp.generation);
      if (state_stamp.generation_changed)
        RCLCPP_WARN(get_logger(), "simulator time rollback; started generation %s",
                    state_stamp.generation.c_str());
      timeline_initialized_ = true;
    }

    RobotState state;
    state.header.stamp = stamp;
    state.header.frame_id = "world";
    state.schema_version = "1.0";
    state.model_identity = kModelIdentity;
    state.generation = state_stamp.generation;
    state.sequence = state_stamp.sequence;
    state.origin = RobotState::ORIGIN_GROUND_TRUTH;
    state.lifecycle = RobotState::LIFECYCLE_RUNNING;
    state.backend_tick = low_state.tick();
    state.base_frame = "base_link";
    state.imu_frame = "imu_link";
    state.has_root_pose = false;
    state.root_pose.orientation.w = 1.0;
    state.has_root_twist = false;
    state.joints.header = state.header;
    state.joints.name.reserve(robotsim_g1_adapter::kG1JointNames.size());
    state.joints.position.reserve(robotsim_g1_adapter::kG1JointNames.size());
    state.joints.velocity.reserve(robotsim_g1_adapter::kG1JointNames.size());
    state.joints.effort.reserve(robotsim_g1_adapter::kG1JointNames.size());
    for (std::size_t index = 0; index < robotsim_g1_adapter::kG1JointNames.size(); ++index) {
      state.joints.name.emplace_back(robotsim_g1_adapter::kG1JointNames[index]);
      state.joints.position.push_back(low_state.motor_state()[index].q());
      state.joints.velocity.push_back(low_state.motor_state()[index].dq());
      state.joints.effort.push_back(low_state.motor_state()[index].tau_est());
    }

    state.imu.header = state.header;
    state.imu.header.frame_id = "imu_link";
    const auto& imu = low_state.imu_state();
    const auto& q = imu.quaternion();
    const double q_norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (q_norm > 1e-9) {
      state.imu.orientation.w = q[0] / q_norm;
      state.imu.orientation.x = q[1] / q_norm;
      state.imu.orientation.y = q[2] / q_norm;
      state.imu.orientation.z = q[3] / q_norm;
    } else {
      state.imu.orientation_covariance[0] = -1.0;
    }
    state.imu.angular_velocity.x = imu.gyroscope()[0];
    state.imu.angular_velocity.y = imu.gyroscope()[1];
    state.imu.angular_velocity.z = imu.gyroscope()[2];
    state.imu.linear_acceleration.x = imu.accelerometer()[0];
    state.imu.linear_acceleration.y = imu.accelerometer()[1];
    state.imu.linear_acceleration.z = imu.accelerometer()[2];

    if (first_state)
      RCLCPP_INFO(get_logger(), "first state generation=%s sequence=%llu",
                  state.generation.c_str(),
                  static_cast<unsigned long long>(state.sequence));

    rosgraph_msgs::msg::Clock clock;
    clock.clock = stamp;
    clock_publisher_->publish(clock);
    state_publisher_->publish(state);
  }

  void publishCommand() {
    LowState low_state;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!have_low_state_) return;
      low_state = latest_low_state_;
    }

    std::array<double, robotsim_g1_adapter::kG1JointNames.size()> efforts{};
    const auto latest = command_buffer_->current(SteadyClock::now());
    if (latest) {
      try {
        efforts = robotsim_g1_adapter::toSdkOrder(latest->joint_names, latest->effort_nm);
      } catch (const std::exception& error) {
        RCLCPP_ERROR(get_logger(), "invalid named G1 command: %s", error.what());
      }
    }

    LowCmd command;
    command.mode_pr() = 0;
    command.mode_machine() = low_state.mode_machine();
    for (std::size_t index = 0; index < robotsim_g1_adapter::kG1JointNames.size(); ++index) {
      auto& motor = command.motor_cmd()[index];
      motor.mode() = 1;
      motor.q() = 0.0f;
      motor.dq() = 0.0f;
      motor.kp() = 0.0f;
      motor.kd() = 0.0f;
      motor.tau() = static_cast<float>(efforts[index]);
    }
    command.crc() = crc32_core(reinterpret_cast<uint32_t*>(&command),
                               (sizeof(LowCmd) >> 2) - 1);
    if (!low_command_publisher_->Write(command, 0))
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "SDK2 LowCmd write failed");
  }

  std::mutex state_mutex_;
  LowState latest_low_state_;
  bool have_low_state_{false};
  uint64_t state_revision_{0};
  uint64_t published_state_revision_{0};
  robotsim_core::StateTimeline timeline_;
  bool timeline_initialized_{false};
  std::unique_ptr<robotsim_core::LatestCommandBuffer> command_buffer_;
  std::unique_ptr<unitree::robot::ChannelSubscriber<LowState>> low_state_subscriber_;
  std::unique_ptr<unitree::robot::ChannelPublisher<LowCmd>> low_command_publisher_;
  rclcpp::Publisher<RobotState>::SharedPtr state_publisher_;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_publisher_;
  rclcpp::Subscription<RobotCommand>::SharedPtr command_subscription_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_;
  rclcpp::TimerBase::SharedPtr state_timer_;
  rclcpp::TimerBase::SharedPtr command_timer_;
};

void requireIsolatedRosEnvironment() {
  const char* domain = std::getenv("ROS_DOMAIN_ID");
  const char* local_only = std::getenv("ROS_LOCALHOST_ONLY");
  if (domain == nullptr || std::string(domain) != "73")
    throw std::invalid_argument("ROS_DOMAIN_ID must be 73 for isolated simulator testing");
  if (local_only == nullptr || std::string(local_only) != "1")
    throw std::invalid_argument("ROS_LOCALHOST_ONLY must be 1");
  if (std::getenv("CYCLONEDDS_URI") != nullptr)
    throw std::invalid_argument("CYCLONEDDS_URI is not supported by the isolated adapter");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    requireIsolatedRosEnvironment();
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<G1Ros2Adapter>());
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "g1_ros2_adapter: " << error.what() << '\n';
    if (rclcpp::ok()) rclcpp::shutdown();
    return 2;
  }
}
