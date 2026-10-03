#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace robotsim_g1_adapter {

inline constexpr std::array<const char*, 29> kG1JointNames{
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint"};

inline std::array<double, kG1JointNames.size()> toSdkOrder(
    const std::vector<std::string>& names, const std::vector<double>& values) {
  if (names.size() != kG1JointNames.size() || values.size() != kG1JointNames.size())
    throw std::invalid_argument("G1 command must name all 29 joints");
  std::array<double, kG1JointNames.size()> ordered{};
  std::array<bool, kG1JointNames.size()> seen{};
  for (std::size_t input = 0; input < names.size(); ++input) {
    std::size_t sdk_index = kG1JointNames.size();
    for (std::size_t candidate = 0; candidate < kG1JointNames.size(); ++candidate) {
      if (names[input] == kG1JointNames[candidate]) {
        sdk_index = candidate;
        break;
      }
    }
    if (sdk_index == kG1JointNames.size()) throw std::invalid_argument("unknown G1 joint name");
    if (seen[sdk_index]) throw std::invalid_argument("duplicate G1 joint name");
    seen[sdk_index] = true;
    ordered[sdk_index] = values[input];
  }
  for (const bool present : seen)
    if (!present) throw std::invalid_argument("missing G1 joint name");
  return ordered;
}

}  // namespace robotsim_g1_adapter
