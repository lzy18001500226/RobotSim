#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using unitree_hg::msg::dds_::LowCmd_;
using unitree_hg::msg::dds_::LowState_;

struct Options {
  int domain_id = 73;
  int seconds = 20;
  int publish_hz = 100;
  int phase_ms = 500;
  int joint = 18;
  int state_timeout_ms = 750;
  float tau_nm = 0.10f;
  std::string interface = "lo";
  std::string output_prefix = "/tmp/g1-lowcmd-roundtrip";
};

struct StateSample {
  uint64_t sequence;
  uint64_t monotonic_ns;
  uint64_t wall_ns;
  uint32_t tick;
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
  recorder->last_state_ns.store(now);
  if (!crc_valid) recorder->bad_crc.fetch_add(1);

  StateSample sample{sequence, now, wallNs(), message.tick(),
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

void printPercentiles(const std::string& label, const std::vector<double>& values) {
  const auto result = percentiles(values);
  if (result.empty()) {
    std::cout << label << "_ms_p50/p95/p99=insufficient\n";
    return;
  }
  std::cout << label << "_ms_p50=" << result[0] << " p95=" << result[1]
            << " p99=" << result[2] << '\n';
}

void writeCsv(const Options& options, const std::vector<StateSample>& states,
              const std::vector<CommandSample>& commands) {
  const std::filesystem::path prefix(options.output_prefix);
  if (!prefix.parent_path().empty())
    std::filesystem::create_directories(prefix.parent_path());

  std::ofstream state_file(prefix.string() + ".lowstate.csv");
  std::ofstream command_file(prefix.string() + ".lowcmd.csv");
  if (!state_file || !command_file) throw std::runtime_error("could not open CSV output");

  state_file << "sequence,monotonic_ns,wall_ns,sim_tick,mode_machine,q,dq,tau_est,crc_valid\n";
  for (const auto& state : states) {
    state_file << state.sequence << ',' << state.monotonic_ns << ',' << state.wall_ns
               << ',' << state.tick << ',' << static_cast<int>(state.mode_machine)
               << ',' << std::setprecision(9) << state.q << ',' << state.dq << ','
               << state.tau_est << ','
               << (state.crc_valid ? 1 : 0) << '\n';
  }

  command_file << "sequence,monotonic_ns,wall_ns,phase,tau_nm,write_ok\n";
  for (const auto& command : commands) {
    command_file << command.sequence << ',' << command.monotonic_ns << ','
                 << command.wall_ns << ',' << command.phase << ','
                 << std::setprecision(9) << command.tau_nm << ','
                 << (command.write_ok ? 1 : 0) << '\n';
  }
}

int run(const Options& options) {
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
  const auto start = Clock::now();
  const auto end = start + std::chrono::seconds(options.seconds);
  const auto period = std::chrono::nanoseconds(1'000'000'000LL / options.publish_hz);
  std::vector<CommandSample> commands;
  commands.reserve(static_cast<size_t>(options.seconds) * options.publish_hz + 1);
  uint64_t sequence = 0;
  uint64_t late_publishes = 0;
  bool state_timeout = false;
  auto next = start;

  std::cout << std::fixed << std::setprecision(3)
            << "run_start_utc=" << wallUtc() << " domain=" << options.domain_id
            << " interface=" << options.interface << " joint=" << options.joint
            << " baseline_q=" << baseline.q << " tau_nm_abs=" << options.tau_nm
            << " publish_hz_target=" << options.publish_hz
            << " phase_ms=" << options.phase_ms << '\n';

  while (Clock::now() < end) {
    const auto now = Clock::now();
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

    const uint64_t tx_ns = monotonicNs();
    const uint64_t tx_wall_ns = wallNs();
    const bool write_ok = publisher.Write(command, 0);
    commands.push_back(CommandSample{++sequence, tx_ns, tx_wall_ns, phase,
                                     sign * options.tau_nm, write_ok});
    if (Clock::now() > next + period) ++late_publishes;
    next += period;
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
  writeCsv(options, states, commands);

  std::vector<double> state_intervals_ms;
  std::vector<double> state_jitter_ms;
  std::vector<double> command_intervals_ms;
  std::vector<double> command_jitter_ms;
  uint64_t tick_gaps = 0;
  uint64_t repeated_ticks = 0;
  uint64_t simulator_ticks = 0;
  uint64_t late_states = 0;
  uint64_t late_commands = 0;
  for (size_t i = 1; i < states.size(); ++i) {
    const double interval = (states[i].monotonic_ns - states[i - 1].monotonic_ns) / 1e6;
    state_intervals_ms.push_back(interval);
    state_jitter_ms.push_back(std::abs(interval - 1.0));
    if (interval > 1.5) ++late_states;
    const uint32_t tick_delta = states[i].tick - states[i - 1].tick;
    simulator_ticks += tick_delta;
    if (tick_delta == 0) ++repeated_ticks;
    else if (tick_delta > 1) tick_gaps += tick_delta - 1;
  }
  for (size_t i = 1; i < commands.size(); ++i) {
    const double interval = (commands[i].monotonic_ns - commands[i - 1].monotonic_ns) / 1e6;
    command_intervals_ms.push_back(interval);
    const double expected = 1000.0 / options.publish_hz;
    command_jitter_ms.push_back(std::abs(interval - expected));
    if (interval > expected * 1.5) ++late_commands;
  }

  std::vector<double> response_ms;
  uint64_t response_missed = 0;
  const double threshold = std::max(0.005, options.tau_nm * 0.5);
  int previous_phase = -1;
  for (size_t i = 0; i < commands.size(); ++i) {
    const auto& command = commands[i];
    if (!command.write_ok || command.phase == previous_phase) continue;
    previous_phase = command.phase;
    const auto before = std::find_if(states.rbegin(), states.rend(),
                                     [&command](const StateSample& state) {
                                       return state.monotonic_ns <= command.monotonic_ns && state.crc_valid;
                                     });
    if (before == states.rend()) {
      ++response_missed;
      continue;
    }
    uint64_t limit_ns = UINT64_MAX;
    for (size_t j = i + 1; j < commands.size(); ++j) {
      if (commands[j].phase != command.phase) {
        limit_ns = commands[j].monotonic_ns;
        break;
      }
    }
    bool observed = false;
    const float direction = command.tau_nm >= 0.0f ? 1.0f : -1.0f;
    for (const auto& state : states) {
      if (state.monotonic_ns < command.monotonic_ns || state.monotonic_ns >= limit_ns ||
          !state.crc_valid) continue;
      if (direction * state.tau_est >= threshold) {
        response_ms.push_back((state.monotonic_ns - command.monotonic_ns) / 1e6);
        observed = true;
        break;
      }
    }
    if (!observed) ++response_missed;
  }

  const double state_seconds = states.size() > 1
      ? (states.back().monotonic_ns - states.front().monotonic_ns) / 1e9 : 0.0;
  const double command_seconds = commands.size() > 1
      ? (commands.back().monotonic_ns - commands.front().monotonic_ns) / 1e9 : 0.0;
  float min_q = baseline.q;
  float max_q = baseline.q;
  float min_tau = baseline.tau_est;
  float max_tau = baseline.tau_est;
  for (const auto& state : states) {
    min_q = std::min(min_q, state.q);
    max_q = std::max(max_q, state.q);
    min_tau = std::min(min_tau, state.tau_est);
    max_tau = std::max(max_tau, state.tau_est);
  }
  const auto state_jitter = percentiles(state_jitter_ms);
  const auto command_jitter = percentiles(command_jitter_ms);

  std::cout << "run_start_wall_ns=" << started_wall_ns
            << " state_samples=" << states.size()
            << " received=" << recorder.received.load()
            << " crc_bad=" << recorder.bad_crc.load()
            << " capture_dropped=" << recorder.capture_dropped.load()
            << " state_rate_hz=" << (state_seconds > 0 ? (states.size() - 1) / state_seconds : 0)
            << " simulator_tick_rate_hz=" << (state_seconds > 0 ? simulator_ticks / state_seconds : 0)
            << " sim_tick_gaps=" << tick_gaps << " repeated_ticks=" << repeated_ticks
            << " late_state_intervals_gt_1_5ms=" << late_states << '\n'
            << "command_samples=" << commands.size()
            << " write_failures=" << std::count_if(commands.begin(), commands.end(),
                                                       [](const CommandSample& c) { return !c.write_ok; })
            << " command_rate_hz=" << (command_seconds > 0 ? (commands.size() - 1) / command_seconds : 0)
            << " late_command_intervals=" << late_commands
            << " late_publish_calls=" << late_publishes << '\n'
            << "selected_joint_q_min=" << min_q << " q_max=" << max_q
            << " q_span=" << (max_q - min_q)
            << " tau_est_min=" << min_tau << " tau_est_max=" << max_tau
            << " response_samples=" << response_ms.size()
            << " response_missed_phases=" << response_missed
            << " torque_response_threshold_nm=" << threshold << '\n'
            << "state_interval_ms_p50/p95/p99=";
  const auto si = percentiles(state_intervals_ms);
  if (si.empty()) std::cout << "insufficient\n";
  else std::cout << si[0] << '/' << si[1] << '/' << si[2] << '\n';
  printPercentiles("state_abs_jitter_from_1ms", state_jitter_ms);
  const auto ci = percentiles(command_intervals_ms);
  std::cout << "command_interval_ms_p50/p95/p99=";
  if (ci.empty()) std::cout << "insufficient\n";
  else std::cout << ci[0] << '/' << ci[1] << '/' << ci[2] << '\n';
  printPercentiles("command_abs_jitter_from_target", command_jitter_ms);
  printPercentiles("lowcmd_to_lowstate_tau_est", response_ms);
  std::cout << "state_abs_jitter_max_ms="
            << (state_jitter_ms.empty() ? 0.0 : *std::max_element(state_jitter_ms.begin(), state_jitter_ms.end()))
            << " command_abs_jitter_max_ms="
            << (command_jitter_ms.empty() ? 0.0 : *std::max_element(command_jitter_ms.begin(), command_jitter_ms.end()))
            << '\n'
            << "csv_lowstate=" << options.output_prefix << ".lowstate.csv\n"
            << "csv_lowcmd=" << options.output_prefix << ".lowcmd.csv\n";

  const uint64_t valid_states = std::count_if(states.begin(), states.end(),
                                             [](const StateSample& state) { return state.crc_valid; });
  const uint64_t successful_writes = std::count_if(commands.begin(), commands.end(),
                                                   [](const CommandSample& command) { return command.write_ok; });
  const bool pass = !state_timeout && valid_states >= 100 && recorder.bad_crc.load() == 0 &&
                    successful_writes > 0 && response_ms.size() >= 2 &&
                    max_tau - min_tau >= threshold * 2;
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
