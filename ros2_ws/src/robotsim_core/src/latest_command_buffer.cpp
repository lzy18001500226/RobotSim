#include "robotsim_core/latest_command_buffer.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace robotsim_core {

LatestCommandBuffer::LatestCommandBuffer(std::string model_identity, std::size_t joint_count,
                                         double max_effort_nm,
                                         std::chrono::milliseconds timeout)
    : model_identity_(std::move(model_identity)), joint_count_(joint_count),
      max_effort_nm_(max_effort_nm), timeout_(timeout) {
  if (model_identity_.empty() || joint_count_ == 0 || !std::isfinite(max_effort_nm_) ||
      max_effort_nm_ <= 0.0 ||
      timeout_ <= std::chrono::milliseconds::zero())
    throw std::invalid_argument("command limits must be positive");
}

bool LatestCommandBuffer::accept(
    const robotsim_interfaces::msg::RobotCommand& command,
    const std::string& active_generation, Clock::time_point received_at,
    std::string* reason) {
  const auto reject = [reason](const char* message) {
    if (reason) *reason = message;
    return false;
  };
  if (command.schema_version != "1.0") return reject("unsupported schema_version");
  if (command.model_identity != model_identity_) return reject("model identity mismatch");
  if (command.generation != active_generation || active_generation != generation_)
    return reject("generation mismatch");
  if (command.source_id.empty()) return reject("source_id is required");
  if (!source_id_.empty() && command.source_id != source_id_)
    return reject("only one command source is configured");
  if (command.command_type != command.COMMAND_TORQUE) return reject("unsupported command type");
  if (command.joint_names.size() != joint_count_ || command.effort_nm.size() != joint_count_)
    return reject("joint vector size mismatch");
  for (const double value : command.effort_nm) {
    if (!std::isfinite(value) || std::abs(value) > max_effort_nm_)
      return reject("effort is non-finite or outside configured limit");
  }
  if (!has_sequence_ && command.sequence != 0)
    return reject("first command sequence must be zero");
  if (has_sequence_ && command.sequence <= last_sequence_)
    return reject("command sequence is not increasing");
  if (command_) ++coalesced_;
  command_ = command;
  source_id_ = command.source_id;
  received_at_ = received_at;
  last_sequence_ = command.sequence;
  has_sequence_ = true;
  if (reason) reason->clear();
  return true;
}

std::optional<robotsim_interfaces::msg::RobotCommand> LatestCommandBuffer::current(
    Clock::time_point now) const {
  if (!command_ || now - received_at_ > timeout_) return std::nullopt;
  return command_;
}

void LatestCommandBuffer::resetGeneration(const std::string& active_generation) {
  generation_ = active_generation;
  command_.reset();
  source_id_.clear();
  has_sequence_ = false;
  last_sequence_ = 0;
}

}  // namespace robotsim_core
