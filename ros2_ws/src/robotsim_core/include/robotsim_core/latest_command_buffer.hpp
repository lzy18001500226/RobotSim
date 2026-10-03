#pragma once

#include <robotsim_interfaces/msg/robot_command.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace robotsim_core {

class LatestCommandBuffer {
 public:
  using Clock = std::chrono::steady_clock;
  LatestCommandBuffer(std::string model_identity, std::size_t joint_count, double max_effort_nm,
                     std::chrono::milliseconds timeout);

  bool accept(const robotsim_interfaces::msg::RobotCommand& command,
              const std::string& active_generation, Clock::time_point received_at,
              std::string* reason = nullptr);
  std::optional<robotsim_interfaces::msg::RobotCommand> current(Clock::time_point now) const;
  void resetGeneration(const std::string& active_generation);
  uint64_t coalesced() const noexcept { return coalesced_; }

 private:
  std::string model_identity_;
  std::size_t joint_count_;
  double max_effort_nm_;
  std::chrono::milliseconds timeout_;
  std::string generation_;
  std::string source_id_;
  std::optional<robotsim_interfaces::msg::RobotCommand> command_;
  Clock::time_point received_at_{};
  uint64_t last_sequence_{0};
  uint64_t coalesced_{0};
  bool has_sequence_{false};
};

}  // namespace robotsim_core
