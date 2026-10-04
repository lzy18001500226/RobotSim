#include "robotsim_core/state_timeline.hpp"

#include <limits>
#include <random>
#include <stdexcept>

namespace robotsim_core {

std::string StateTimeline::newGeneration() {
  std::random_device random;
  return "robotsim-" + std::to_string((static_cast<uint64_t>(random()) << 32) ^ random());
}

StateTimeline::StateTimeline() : generation_(newGeneration()) {}

StateStamp StateTimeline::next(int64_t sim_time_ns, bool force_new_generation) {
  if (sim_time_ns < 0) throw std::invalid_argument("simulation time must be nonnegative");
  bool changed = false;
  if (!initialized_) {
    initialized_ = true;
    sequence_ = 0;
  } else if (force_new_generation || sim_time_ns < last_sim_time_ns_) {
    generation_ = newGeneration();
    sequence_ = 0;
    changed = true;
  } else {
    if (sequence_ == std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("state sequence exhausted within generation");
    ++sequence_;
  }
  last_sim_time_ns_ = sim_time_ns;
  return {generation_, sequence_, changed};
}

}  // namespace robotsim_core
