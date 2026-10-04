#pragma once

#include <builtin_interfaces/msg/time.hpp>

#include <cstdint>

namespace robotsim_g1_adapter {

inline builtin_interfaces::msg::Time tickMillisecondsToRosTime(uint32_t tick) {
  const uint64_t ns = static_cast<uint64_t>(tick) * 1000000ULL;
  builtin_interfaces::msg::Time result;
  result.sec = static_cast<int32_t>(ns / 1000000000ULL);
  result.nanosec = static_cast<uint32_t>(ns % 1000000000ULL);
  return result;
}

}  // namespace robotsim_g1_adapter
