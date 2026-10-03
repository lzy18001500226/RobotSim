#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <ifaddrs.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <net/if.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;
using unitree_hg::msg::dds_::LowCmd_;
using unitree_hg::msg::dds_::LowState_;

constexpr int kIsolatedDdsDomain = 73;
constexpr const char* kIsolatedDdsInterface = "lo";

struct Options {
  int domain_id = kIsolatedDdsDomain;
  int seconds = 20;
  int publish_hz = 100;
  int phase_ms = 500;
  int joint = 18;
  int state_timeout_ms = 750;
  float tau_nm = 0.10f;
  std::string interface = kIsolatedDdsInterface;
  std::string output_prefix;
};

struct StateSample {
  uint64_t sequence;
  uint64_t monotonic_ns;
  uint64_t wall_ns;
  uint32_t tick;
  uint64_t tick_generation;
  uint8_t mode_machine;
  float q;
  float dq;
  float tau_est;
  bool crc_valid;
};

struct CommandSample {
  uint64_t sequence;
  uint64_t monotonic_ns;
  uint64_t wall_ns;
  int phase;
  float tau_nm;
  bool write_ok;
  uint64_t deadline_lateness_ns;
  uint64_t missed_slots_after;
};

struct Recorder {
  explicit Recorder(size_t capacity) { states.reserve(capacity); }

  std::mutex mutex;
  std::condition_variable ready;
  std::vector<StateSample> states;
  size_t capacity = 0;
  std::atomic<uint64_t> received{0};
  std::atomic<uint64_t> bad_crc{0};
  std::atomic<uint64_t> capture_dropped{0};
  std::atomic<uint64_t> last_state_ns{0};
};

uint64_t monotonicNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
      .count();
}

uint64_t wallNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string wallUtc() {
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&seconds, &utc);
  const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch())
                          .count() % 1000;
  std::ostringstream out;
  out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.'
      << std::setw(3) << std::setfill('0') << millis << 'Z';
  return out.str();
}

std::string makeDefaultOutputPrefix() {
  const auto filename = "g1-lowcmd-roundtrip-" + std::to_string(wallNs()) +
                        "-" + std::to_string(static_cast<long long>(::getpid()));
  return (std::filesystem::temp_directory_path() / filename).string();
}

int parseInt(const std::string& value, const std::string& name) {
  size_t used = 0;
  const int result = std::stoi(value, &used);
  if (used != value.size()) throw std::invalid_argument("invalid " + name);
  return result;
}

float parseFloat(const std::string& value, const std::string& name) {
  size_t used = 0;
  const float result = std::stof(value, &used);
  if (used != value.size() || !std::isfinite(result))
    throw std::invalid_argument("invalid " + name);
  return result;
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (i + 1 >= argc) throw std::invalid_argument("missing option value");
    const std::string name = argv[i++];
    const std::string value = argv[i];
    if (name == "--domain") options.domain_id = parseInt(value, name);
    else if (name == "--interface") options.interface = value;
    else if (name == "--seconds") options.seconds = parseInt(value, name);
    else if (name == "--publish-hz") options.publish_hz = parseInt(value, name);
    else if (name == "--phase-ms") options.phase_ms = parseInt(value, name);
    else if (name == "--joint") options.joint = parseInt(value, name);
    else if (name == "--tau-nm") options.tau_nm = parseFloat(value, name);
    else if (name == "--state-timeout-ms") options.state_timeout_ms = parseInt(value, name);
    else if (name == "--output-prefix") options.output_prefix = value;
    else throw std::invalid_argument("unknown option: " + name);
  }

  if (options.domain_id < 0 || options.domain_id > 232 || options.interface.empty() ||
      options.seconds < 1 || options.seconds > 120 || options.publish_hz < 1 ||
      options.publish_hz > 250 || options.phase_ms < 50 || options.phase_ms > 2000 ||
      options.joint < 0 || options.joint >= 29 || options.tau_nm <= 0.0f ||
      options.tau_nm > 0.2f || options.state_timeout_ms < 100 ||
      options.state_timeout_ms > 5000) {
    throw std::invalid_argument("option outside the probe's bounded test limits");
  }
  if (options.domain_id != kIsolatedDdsDomain ||
      options.interface != kIsolatedDdsInterface) {
    throw std::invalid_argument(
        "Issue #9 probe only supports isolated DDS domain 73 on interface lo");
  }
  if (options.output_prefix.empty()) options.output_prefix = makeDefaultOutputPrefix();
  return options;
}

void receiveLowState(const void* raw, Recorder* recorder, int joint) {
  const auto* incoming = static_cast<const LowState_*>(raw);
  LowState_ message = *incoming;
  const bool crc_valid =
      message.crc() == crc32_core(reinterpret_cast<uint32_t*>(&message),
                                  (sizeof(LowState_) >> 2) - 1);
  const uint64_t now = monotonicNs();
  const uint64_t sequence = recorder->received.fetch_add(1) + 1;
  if (crc_valid) recorder->last_state_ns.store(now);
  if (!crc_valid) recorder->bad_crc.fetch_add(1);

  StateSample sample{sequence, now, wallNs(), message.tick(), 0,
                     message.mode_machine(), message.motor_state()[joint].q(),
                     message.motor_state()[joint].dq(),
                     message.motor_state()[joint].tau_est(), crc_valid};
  {
    std::lock_guard<std::mutex> lock(recorder->mutex);
    if (recorder->states.size() < recorder->capacity) {
      recorder->states.push_back(sample);
    } else {
      recorder->capture_dropped.fetch_add(1);
    }
  }
  recorder->ready.notify_one();
}

std::vector<double> percentiles(const std::vector<double>& values) {
  if (values.empty()) return {};
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  auto at = [&sorted](double p) {
    const size_t index = static_cast<size_t>(std::ceil(p * sorted.size())) - 1;
    return sorted[std::min(index, sorted.size() - 1)];
  };
  return {at(0.50), at(0.95), at(0.99)};
}

void requireEvidencePathsAvailable(const Options& options) {
  const std::filesystem::path prefix(options.output_prefix);
  const std::vector<std::filesystem::path> paths{
      prefix.string() + ".lowstate.csv",
      prefix.string() + ".lowcmd.csv",
      prefix.string() + ".summary.json"};
  for (const auto& path : paths) {
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error) {
      throw std::runtime_error("could not inspect evidence path " + path.string() +
                               ": " + error.message());
    }
    if (exists) {
      throw std::runtime_error("evidence path already exists; choose a new --output-prefix: " +
                               path.string());
    }
  }
}

void requireEnvironmentValue(const char* name, const char* expected) {
  const char* value = std::getenv(name);
  if (value != nullptr && value != std::string(expected)) {
    throw std::invalid_argument(std::string(name) +
                                " must be unset or equal to " + expected);
  }
}

void requireIsolatedDdsConfiguration(const Options& options) {
  if (options.domain_id != kIsolatedDdsDomain ||
      options.interface != kIsolatedDdsInterface) {
    throw std::invalid_argument(
        "Issue #9 probe only supports isolated DDS domain 73 on interface lo");
  }
  requireEnvironmentValue("ISSUE9_DOMAIN_ID", "73");
  requireEnvironmentValue("ISSUE9_DDS_INTERFACE", "lo");
  requireEnvironmentValue("CYCLONEDDS_DOMAIN_ID", "73");
  requireEnvironmentValue("ROS_DOMAIN_ID", "73");
  if (std::getenv("CYCLONEDDS_URI") != nullptr) {
    throw std::invalid_argument(
        "CYCLONEDDS_URI overrides are unsupported by the isolated Issue #9 probe");
  }

  struct ifaddrs* interfaces = nullptr;
  if (::getifaddrs(&interfaces) != 0) {
    throw std::runtime_error("could not inspect local network interfaces");
  }
  bool loopback_is_up = false;
  for (const struct ifaddrs* interface = interfaces; interface != nullptr;
       interface = interface->ifa_next) {
    if (interface->ifa_name != nullptr &&
        options.interface == interface->ifa_name &&
        (interface->ifa_flags & IFF_LOOPBACK) != 0 &&
        (interface->ifa_flags & IFF_UP) != 0) {
      loopback_is_up = true;
      break;
    }
  }
  ::freeifaddrs(interfaces);
  if (!loopback_is_up) {
    throw std::invalid_argument(
        "Issue #9 probe requires the active OS loopback interface named lo");
  }
}

void printPercentiles(const std::string& label, const std::vector<double>& values) {
  const auto result = percentiles(values);
  if (result.empty()) {
    std::cout << label << "_ms_p50/p95/p99=insufficient\n";
    return;
  }
  std::cout << label << "_ms_p50=" << result[0] << " p95=" << result[1]
            << " p99=" << result[2] << '\n';
}

void writeExclusiveFiles(
    const std::vector<std::pair<std::filesystem::path, std::string>>& files) {
  std::vector<int> descriptors(files.size(), -1);
  std::vector<bool> created(files.size(), false);
  auto cleanup = [&] {
    for (int& descriptor : descriptors) {
      if (descriptor >= 0) {
        ::close(descriptor);
        descriptor = -1;
      }
    }
    for (size_t i = 0; i < files.size(); ++i) {
      if (created[i]) ::unlink(files[i].first.c_str());
    }
  };

  try {
    for (size_t i = 0; i < files.size(); ++i) {
      descriptors[i] = ::open(files[i].first.c_str(),
                              O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
      if (descriptors[i] < 0) {
        const int error = errno;
        throw std::runtime_error(
            "could not create evidence file " + files[i].first.string() +
            " exclusively: " + std::strerror(error));
      }
      created[i] = true;
    }

    for (size_t i = 0; i < files.size(); ++i) {
      const std::string& contents = files[i].second;
      size_t offset = 0;
      while (offset < contents.size()) {
        const ssize_t written = ::write(descriptors[i], contents.data() + offset,
                                        contents.size() - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
          const int error = errno;
          throw std::runtime_error("could not write evidence file " +
                                   files[i].first.string() + ": " +
                                   std::strerror(error));
        }
        offset += static_cast<size_t>(written);
      }
      if (::fsync(descriptors[i]) != 0) {
        const int error = errno;
        throw std::runtime_error("could not flush evidence file " +
                                 files[i].first.string() + ": " +
                                 std::strerror(error));
      }
    }

    for (size_t i = 0; i < files.size(); ++i) {
      const int descriptor = descriptors[i];
      descriptors[i] = -1;
      if (::close(descriptor) != 0) {
        const int error = errno;
        throw std::runtime_error("could not close evidence file " +
                                 files[i].first.string() + ": " +
                                 std::strerror(error));
      }
    }
  } catch (...) {
    cleanup();
    throw;
  }
}

void writeEvidence(const Options& options, std::vector<StateSample>& states,
                   const std::vector<CommandSample>& commands,
                   const std::string& summary_json) {
  const std::filesystem::path prefix(options.output_prefix);
  if (!prefix.parent_path().empty())
    std::filesystem::create_directories(prefix.parent_path());

  std::ostringstream state_file;
  std::ostringstream command_file;

  state_file << "sequence,monotonic_ns,wall_ns,sim_tick,tick_generation,mode_machine,q,dq,tau_est,crc_valid\n";
  for (const auto& state : states) {
    state_file << state.sequence << ',' << state.monotonic_ns << ',' << state.wall_ns
               << ',' << state.tick << ',' << state.tick_generation << ','
               << static_cast<int>(state.mode_machine)
               << ',' << std::setprecision(9) << state.q << ',' << state.dq << ','
               << state.tau_est << ','
               << (state.crc_valid ? 1 : 0) << '\n';
  }

  command_file << "sequence,monotonic_ns,wall_ns,phase,tau_nm,write_ok,deadline_lateness_ns,missed_slots_after\n";
  for (const auto& command : commands) {
    command_file << command.sequence << ',' << command.monotonic_ns << ','
                 << command.wall_ns << ',' << command.phase << ','
                 << std::setprecision(9) << command.tau_nm << ','
                 << (command.write_ok ? 1 : 0) << ','
                 << command.deadline_lateness_ns << ','
                 << command.missed_slots_after << '\n';
  }

  const std::vector<std::pair<std::filesystem::path, std::string>> files{
      {prefix.string() + ".lowstate.csv", state_file.str()},
      {prefix.string() + ".lowcmd.csv", command_file.str()},
      {prefix.string() + ".summary.json", summary_json}};
  writeExclusiveFiles(files);
}

int run(const Options& options) {
  requireIsolatedDdsConfiguration(options);
  requireEvidencePathsAvailable(options);

  constexpr size_t kMaxCapturedStates = 1'000'000;
  const size_t capacity = std::min(
      kMaxCapturedStates, static_cast<size_t>(options.seconds) * 5000 + 10000);
  Recorder recorder(capacity);
  recorder.capacity = capacity;

  unitree::robot::ChannelFactory::Instance()->Init(options.domain_id, options.interface);
  unitree::robot::ChannelSubscriber<LowState_> subscriber("rt/lowstate");
  subscriber.InitChannel(
      [&recorder, &options](const void* message) {
        receiveLowState(message, &recorder, options.joint);
      },
      1);
  unitree::robot::ChannelPublisher<LowCmd_> publisher("rt/lowcmd");
  publisher.InitChannel();

  {
    std::unique_lock<std::mutex> lock(recorder.mutex);
    if (!recorder.ready.wait_for(lock, std::chrono::seconds(10),
                                 [&recorder] { return !recorder.states.empty(); })) {
      std::cerr << "FAIL: no LowState received within 10 s\n";
      return 2;
    }
  }

  StateSample baseline{};
  {
    std::lock_guard<std::mutex> lock(recorder.mutex);
    const auto found = std::find_if(recorder.states.rbegin(), recorder.states.rend(),
                                    [](const StateSample& state) { return state.crc_valid; });
    if (found == recorder.states.rend()) {
      std::cerr << "FAIL: LowState arrived but no CRC-valid sample was received\n";
      return 2;
    }
    baseline = *found;
  }

  const uint64_t started_wall_ns = wallNs();
  const std::string started_utc = wallUtc();
  const auto start = Clock::now();
  const auto end = start + std::chrono::seconds(options.seconds);
  const uint64_t start_steady_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
  const auto period = std::chrono::nanoseconds(1'000'000'000LL / options.publish_hz);
  const uint64_t duration_ms = static_cast<uint64_t>(options.seconds) * 1000;
  const uint64_t expected_phases =
      (duration_ms + static_cast<uint64_t>(options.phase_ms) - 1) /
      static_cast<uint64_t>(options.phase_ms);
  std::vector<const CommandSample*> first_phase_command(expected_phases, nullptr);
  std::vector<CommandSample> commands;
  commands.reserve(static_cast<size_t>(options.seconds) * options.publish_hz + 1);
  uint64_t sequence = 0;
  uint64_t late_command_starts_gt_period = 0;
  uint64_t missed_command_slots = 0;
  bool state_timeout = false;
  bool state_crc_error = false;
  auto next = start;

  std::cout << std::fixed << std::setprecision(3)
            << "run_start_utc=" << started_utc << " domain=" << options.domain_id
            << " interface=" << options.interface << " joint=" << options.joint
            << " baseline_q=" << baseline.q << " tau_nm_abs=" << options.tau_nm
            << " publish_hz_target=" << options.publish_hz
            << " phase_ms=" << options.phase_ms << '\n';

  while (Clock::now() < end) {
    const auto now = Clock::now();
    if (recorder.bad_crc.load() != 0) {
      state_crc_error = true;
      std::cerr << "FAIL: LowState CRC error; stopping LowCmd publication\n";
      break;
    }
    const auto last_state_ns = recorder.last_state_ns.load();
    const auto age_ns = monotonicNs() - last_state_ns;
    if (last_state_ns == 0 || age_ns >
                                  static_cast<uint64_t>(options.state_timeout_ms) * 1'000'000) {
      state_timeout = true;
      std::cerr << "FAIL: LowState timeout age_ms=" << age_ns / 1'000'000 << '\n';
      break;
    }

    const int phase = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() /
        options.phase_ms);
    const float sign = phase % 2 == 0 ? 1.0f : -1.0f;
    LowCmd_ command;
    command.mode_pr() = 0;
    command.mode_machine() = baseline.mode_machine;
    for (int i = 0; i < 29; ++i) {
      auto& motor = command.motor_cmd()[i];
      motor.mode() = 1;
      motor.q() = baseline.q;
      motor.dq() = 0.0f;
      motor.kp() = 0.0f;
      motor.kd() = 0.0f;
      motor.tau() = 0.0f;
    }
    auto& test_motor = command.motor_cmd()[options.joint];
    test_motor.tau() = sign * options.tau_nm;
    command.crc() = crc32_core(reinterpret_cast<uint32_t*>(&command),
                                (sizeof(LowCmd_) >> 2) - 1);

    const auto send_time = Clock::now();
    const auto lateness = send_time > next
        ? std::chrono::duration_cast<std::chrono::nanoseconds>(send_time - next).count()
        : 0;
    if (lateness > period.count()) ++late_command_starts_gt_period;
    const uint64_t tx_ns = monotonicNs();
    const uint64_t tx_wall_ns = wallNs();
    const bool write_ok = publisher.Write(command, 0);
    commands.push_back(CommandSample{++sequence, tx_ns, tx_wall_ns, phase,
                                     sign * options.tau_nm, write_ok,
                                     static_cast<uint64_t>(lateness), 0});
    if (phase >= 0 && static_cast<uint64_t>(phase) < expected_phases &&
        first_phase_command[phase] == nullptr) {
      first_phase_command[phase] = &commands.back();
    }

    next += period;
    const auto after_write = Clock::now();
    uint64_t skipped_after_write = 0;
    while (next <= after_write) {
      ++missed_command_slots;
      ++skipped_after_write;
      next += period;
    }
    commands.back().missed_slots_after = skipped_after_write;
    std::this_thread::sleep_until(next);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  subscriber.CloseChannel();
  publisher.CloseChannel();

  std::vector<StateSample> states;
  {
    std::lock_guard<std::mutex> lock(recorder.mutex);
    states = recorder.states;
  }

  std::vector<double> state_intervals_ms;
  std::vector<double> state_jitter_ms;
  std::vector<double> command_intervals_ms;
  std::vector<double> command_jitter_ms;
  std::vector<double> command_start_lateness_ms;
  uint64_t tick_gaps = 0;
  uint64_t repeated_ticks = 0;
  uint64_t simulator_ticks = 0;
  uint64_t tick_resets = 0;
  uint64_t tick_elapsed_ns = 0;
  uint64_t late_state_callback_gaps = 0;
  uint64_t late_commands = 0;

  std::vector<size_t> valid_state_indices;
  valid_state_indices.reserve(states.size());
  for (size_t i = 0; i < states.size(); ++i) {
    if (states[i].crc_valid) valid_state_indices.push_back(i);
  }

  size_t previous_valid_index = states.size();
  uint64_t tick_generation = 0;
  for (size_t index = 0; index < states.size(); ++index) {
    auto& state = states[index];
    state.tick_generation = tick_generation;
    if (!state.crc_valid) continue;
    if (previous_valid_index == states.size()) {
      previous_valid_index = index;
      continue;
    }

    const auto& previous = states[previous_valid_index];
    const uint64_t interval_ns = state.monotonic_ns - previous.monotonic_ns;
    const double interval = interval_ns / 1e6;
    state_intervals_ms.push_back(interval);
    state_jitter_ms.push_back(std::abs(interval - 1.0));
    if (interval > 1.5) ++late_state_callback_gaps;

    const bool tick_decreased = state.tick < previous.tick;
    const bool uint32_wrap = previous.tick > 0xF0000000U && state.tick < 0x0FFFFFFFU;
    if (tick_decreased && !uint32_wrap) {
      ++tick_resets;
      ++tick_generation;
      state.tick_generation = tick_generation;
    } else {
      const uint32_t tick_delta = state.tick - previous.tick;
      simulator_ticks += tick_delta;
      tick_elapsed_ns += interval_ns;
      if (tick_delta == 0) ++repeated_ticks;
      else if (tick_delta > 1) tick_gaps += tick_delta - 1;
    }
    previous_valid_index = index;
  }
  for (size_t i = 1; i < commands.size(); ++i) {
    const double interval = (commands[i].monotonic_ns - commands[i - 1].monotonic_ns) / 1e6;
    command_intervals_ms.push_back(interval);
    const double expected = 1000.0 / options.publish_hz;
    command_jitter_ms.push_back(std::abs(interval - expected));
    if (interval > expected * 1.5) ++late_commands;
  }
  for (const auto& command : commands) {
    command_start_lateness_ms.push_back(command.deadline_lateness_ns / 1e6);
  }

  std::vector<double> response_ms;
  uint64_t response_missed = 0;
  const double threshold = std::max(0.005, options.tau_nm * 0.5);
  uint64_t commanded_phases = 0;
  for (uint64_t phase = 0; phase < expected_phases; ++phase) {
    const CommandSample* command = first_phase_command[phase];
    if (command == nullptr) {
      ++response_missed;
      continue;
    }
    ++commanded_phases;
    if (!command->write_ok) {
      ++response_missed;
      continue;
    }
    const auto before = std::find_if(states.rbegin(), states.rend(),
                                     [command](const StateSample& state) {
                                       return state.monotonic_ns <= command->monotonic_ns && state.crc_valid;
                                     });
    if (before == states.rend()) {
      ++response_missed;
      continue;
    }
    const float direction = command->tau_nm >= 0.0f ? 1.0f : -1.0f;
    if (direction * before->tau_est >= threshold) {
      ++response_missed;
      continue;
    }
    const uint64_t phase_end_ns = start_steady_ns +
        std::min(duration_ms, (phase + 1) * static_cast<uint64_t>(options.phase_ms)) * 1'000'000;
    bool observed = false;
    for (const auto& state : states) {
      if (state.monotonic_ns < command->monotonic_ns ||
          state.monotonic_ns >= phase_end_ns ||
          !state.crc_valid) continue;
      if (direction * state.tau_est >= threshold) {
        response_ms.push_back((state.monotonic_ns - command->monotonic_ns) / 1e6);
        observed = true;
        break;
      }
    }
    if (!observed) ++response_missed;
  }

  const double state_seconds = valid_state_indices.size() > 1
      ? (states[valid_state_indices.back()].monotonic_ns -
         states[valid_state_indices.front()].monotonic_ns) / 1e9 : 0.0;
  const double tick_seconds = tick_elapsed_ns / 1e9;
  const double command_seconds = commands.size() > 1
      ? (commands.back().monotonic_ns - commands.front().monotonic_ns) / 1e9 : 0.0;
  float min_q = baseline.q;
  float max_q = baseline.q;
  float min_tau = baseline.tau_est;
  float max_tau = baseline.tau_est;
  for (const size_t index : valid_state_indices) {
    const auto& state = states[index];
    min_q = std::min(min_q, state.q);
    max_q = std::max(max_q, state.q);
    min_tau = std::min(min_tau, state.tau_est);
    max_tau = std::max(max_tau, state.tau_est);
  }
  const auto state_jitter = percentiles(state_jitter_ms);
  const auto command_jitter = percentiles(command_jitter_ms);
  const auto si = percentiles(state_intervals_ms);
  const auto ci = percentiles(command_intervals_ms);
  const auto response_percentiles = percentiles(response_ms);
  const auto command_start_lateness = percentiles(command_start_lateness_ms);

  const uint64_t valid_states = valid_state_indices.size();
  const uint64_t run_end_steady_ns =
      start_steady_ns + duration_ms * 1'000'000;
  const uint64_t valid_states_during_test = std::count_if(
      valid_state_indices.begin(), valid_state_indices.end(),
      [&states, start_steady_ns, run_end_steady_ns](size_t index) {
        return states[index].monotonic_ns >= start_steady_ns &&
               states[index].monotonic_ns < run_end_steady_ns;
      });
  const uint64_t capture_dropped = recorder.capture_dropped.load();
  const uint64_t bad_crc = recorder.bad_crc.load();
  const uint64_t write_failures = std::count_if(
      commands.begin(), commands.end(),
      [](const CommandSample& command) { return !command.write_ok; });
  const uint64_t missing_command_phases = expected_phases - commanded_phases;
  const uint64_t minimum_valid_states = std::max<uint64_t>(100, expected_phases);
  const bool complete_phase_coverage = commanded_phases == expected_phases &&
                                      response_ms.size() == expected_phases;
  const bool pass = !state_timeout && !state_crc_error &&
                    valid_states_during_test >= minimum_valid_states &&
                    bad_crc == 0 && capture_dropped == 0 && write_failures == 0 &&
                    !commands.empty() && complete_phase_coverage &&
                    max_tau - min_tau >= threshold * 2;

  std::ostringstream summary;
  summary << std::setprecision(12)
          << "{\n  \"schema_version\": 1,\n"
          << "  \"result\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
          << "  \"run_start_utc\": \"" << started_utc << "\",\n"
          << "  \"run_start_wall_ns\": \"" << started_wall_ns << "\",\n"
          << "  \"domain_id\": " << options.domain_id << ",\n"
          << "  \"interface\": \"" << options.interface << "\",\n"
          << "  \"duration_seconds\": " << options.seconds << ",\n"
          << "  \"publish_hz_target\": " << options.publish_hz << ",\n"
          << "  \"phase_ms\": " << options.phase_ms << ",\n"
          << "  \"joint\": " << options.joint << ",\n"
          << "  \"tau_nm_abs\": " << options.tau_nm << ",\n"
          << "  \"expected_phases\": " << expected_phases << ",\n"
          << "  \"commanded_phases\": " << commanded_phases << ",\n"
          << "  \"observed_phases\": " << response_ms.size() << ",\n"
          << "  \"missed_phases\": " << response_missed << ",\n"
          << "  \"missing_command_phases\": " << missing_command_phases << ",\n"
          << "  \"command_samples\": " << commands.size() << ",\n"
          << "  \"write_failures\": " << write_failures << ",\n"
          << "  \"missed_command_slots\": " << missed_command_slots << ",\n"
          << "  \"states_received\": " << recorder.received.load() << ",\n"
          << "  \"states_captured\": " << states.size() << ",\n"
          << "  \"crc_valid_states\": " << valid_states << ",\n"
          << "  \"crc_valid_states_during_test\": " << valid_states_during_test << ",\n"
          << "  \"crc_invalid_states\": " << bad_crc << ",\n"
          << "  \"capture_dropped\": " << capture_dropped << ",\n"
          << "  \"state_callback_rate_hz\": "
          << (state_seconds > 0 ? (valid_states - 1) / state_seconds : 0) << ",\n"
          << "  \"simulator_tick_rate_hz\": "
          << (tick_seconds > 0 ? simulator_ticks / tick_seconds : 0) << ",\n"
          << "  \"tick_resets\": " << tick_resets << ",\n"
          << "  \"tick_generation_count\": " << (tick_generation + 1) << ",\n"
          << "  \"tick_gaps\": " << tick_gaps << ",\n"
          << "  \"repeated_ticks\": " << repeated_ticks << ",\n"
          << "  \"state_interval_ms_p50_p95_p99\": ["
          << (si.empty() ? 0.0 : si[0]) << ',' << (si.empty() ? 0.0 : si[1])
          << ',' << (si.empty() ? 0.0 : si[2]) << "],\n"
          << "  \"state_abs_jitter_ms_p50_p95_p99\": ["
          << (state_jitter.empty() ? 0.0 : state_jitter[0]) << ','
          << (state_jitter.empty() ? 0.0 : state_jitter[1]) << ','
          << (state_jitter.empty() ? 0.0 : state_jitter[2]) << "],\n"
          << "  \"command_interval_ms_p50_p95_p99\": ["
          << (ci.empty() ? 0.0 : ci[0]) << ',' << (ci.empty() ? 0.0 : ci[1])
          << ',' << (ci.empty() ? 0.0 : ci[2]) << "],\n"
          << "  \"command_abs_jitter_ms_p50_p95_p99\": ["
          << (command_jitter.empty() ? 0.0 : command_jitter[0]) << ','
          << (command_jitter.empty() ? 0.0 : command_jitter[1]) << ','
          << (command_jitter.empty() ? 0.0 : command_jitter[2]) << "],\n"
          << "  \"command_start_lateness_ms_p50_p95_p99\": ["
          << (command_start_lateness.empty() ? 0.0 : command_start_lateness[0]) << ','
          << (command_start_lateness.empty() ? 0.0 : command_start_lateness[1]) << ','
          << (command_start_lateness.empty() ? 0.0 : command_start_lateness[2]) << "],\n"
          << "  \"callback_arrival_sign_threshold_ms_p50_p95_p99\": ["
          << (response_percentiles.empty() ? 0.0 : response_percentiles[0]) << ','
          << (response_percentiles.empty() ? 0.0 : response_percentiles[1]) << ','
          << (response_percentiles.empty() ? 0.0 : response_percentiles[2]) << "]\n}\n";

  writeEvidence(options, states, commands, summary.str());

  std::cout << "run_start_wall_ns=" << started_wall_ns
            << " state_samples=" << states.size()
            << " received=" << recorder.received.load()
            << " crc_bad=" << bad_crc
            << " capture_dropped=" << capture_dropped
            << " state_rate_hz=" << (state_seconds > 0 ? (valid_states - 1) / state_seconds : 0)
            << " simulator_tick_rate_hz=" << (tick_seconds > 0 ? simulator_ticks / tick_seconds : 0)
            << " sim_tick_gaps=" << tick_gaps << " repeated_ticks=" << repeated_ticks
            << " tick_resets=" << tick_resets
            << " state_callback_gaps_gt_1_5ms=" << late_state_callback_gaps << '\n'
            << "command_samples=" << commands.size()
            << " write_failures=" << write_failures
            << " command_rate_hz=" << (command_seconds > 0 ? (commands.size() - 1) / command_seconds : 0)
            << " command_intervals_gt_1_5x_target=" << late_commands
            << " command_starts_late_gt_one_period=" << late_command_starts_gt_period
            << " missed_command_slots=" << missed_command_slots << '\n'
            << "selected_joint_q_min=" << min_q << " q_max=" << max_q
            << " q_span=" << (max_q - min_q)
            << " tau_est_min=" << min_tau << " tau_est_max=" << max_tau
            << " response_samples=" << response_ms.size()
            << " expected_phases=" << expected_phases
            << " commanded_phases=" << commanded_phases
            << " missing_command_phases=" << missing_command_phases
            << " response_missed_phases=" << response_missed
            << " torque_response_threshold_nm=" << threshold << '\n'
            << "state_interval_ms_p50/p95/p99=";
  if (si.empty()) std::cout << "insufficient\n";
  else std::cout << si[0] << '/' << si[1] << '/' << si[2] << '\n';
  printPercentiles("state_abs_jitter_from_1ms", state_jitter_ms);
  std::cout << "command_interval_ms_p50/p95/p99=";
  if (ci.empty()) std::cout << "insufficient\n";
  else std::cout << ci[0] << '/' << ci[1] << '/' << ci[2] << '\n';
  printPercentiles("command_abs_jitter_from_target", command_jitter_ms);
  printPercentiles("send_to_callback_arrival_sign_threshold_estimate", response_ms);
  std::cout << "state_abs_jitter_max_ms="
            << (state_jitter_ms.empty() ? 0.0 : *std::max_element(state_jitter_ms.begin(), state_jitter_ms.end()))
            << " command_abs_jitter_max_ms="
            << (command_jitter_ms.empty() ? 0.0 : *std::max_element(command_jitter_ms.begin(), command_jitter_ms.end()))
            << '\n'
            << "csv_lowstate=" << options.output_prefix << ".lowstate.csv\n"
            << "csv_lowcmd=" << options.output_prefix << ".lowcmd.csv\n"
            << "summary_json=" << options.output_prefix << ".summary.json\n";

  std::cout << (pass ? "PASS" : "FAIL") << ": bounded LowCmd -> simulator -> LowState check"
            << " (domain=" << options.domain_id << ", joint=" << options.joint << ")\n";
  return pass ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parseOptions(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 2;
  }
}
