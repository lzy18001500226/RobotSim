#pragma once

#include <cstdint>
#include <mutex>

namespace robotsim_core {

class RollbackInhibit {
 public:
  struct Snapshot {
    uint64_t epoch;
    bool pending;
  };

  bool observeTick(uint64_t tick) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool rolled_back = have_tick_ && tick < last_tick_;
    if (rolled_back) ++observed_epoch_;
    last_tick_ = tick;
    have_tick_ = true;
    return rolled_back;
  }

  Snapshot snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {observed_epoch_, observed_epoch_ != processed_epoch_};
  }

  void generationEstablished(uint64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (epoch <= observed_epoch_ && epoch > processed_epoch_)
      processed_epoch_ = epoch;
  }

  // Linearize command acceptance/output against SDK tick intake.
  template <typename Operation>
  void synchronize(Operation&& operation) {
    std::lock_guard<std::mutex> lock(mutex_);
    operation(observed_epoch_ != processed_epoch_);
  }

 private:
  mutable std::mutex mutex_;
  uint64_t last_tick_{0};
  uint64_t observed_epoch_{0};
  uint64_t processed_epoch_{0};
  bool have_tick_{false};
};

}  // namespace robotsim_core
