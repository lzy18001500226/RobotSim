#pragma once

#include <cstdint>
#include <string>

namespace robotsim_core {

struct StateStamp {
  std::string generation;
  uint64_t sequence;
  bool generation_changed;
};

class StateTimeline {
 public:
  StateTimeline();
  StateStamp next(int64_t sim_time_ns, bool force_new_generation = false);
  const std::string& generation() const noexcept { return generation_; }

 private:
  static std::string newGeneration();
  std::string generation_;
  int64_t last_sim_time_ns_{0};
  uint64_t sequence_{0};
  bool initialized_{false};
};

}  // namespace robotsim_core
