#include "robotsim_core/latest_command_buffer.hpp"
#include "robotsim_core/rollback_inhibit.hpp"
#include "robotsim_core/state_timeline.hpp"

#include <gtest/gtest.h>

#include <optional>

using robotsim_core::LatestCommandBuffer;

TEST(StateTimeline, IncrementsSequenceAndChangesGenerationOnRollback) {
  robotsim_core::StateTimeline timeline;
  const auto first = timeline.next(1000000);
  const auto repeat = timeline.next(1000000);
  const auto rollback = timeline.next(0);
  EXPECT_EQ(first.sequence, 0u);
  EXPECT_EQ(repeat.sequence, 1u);
  EXPECT_FALSE(first.generation_changed);
  EXPECT_NE(first.generation, rollback.generation);
  EXPECT_TRUE(rollback.generation_changed);
  EXPECT_EQ(rollback.sequence, 0u);
}

TEST(StateTimeline, ForcedRollbackStartsGenerationEvenWhenLatestTimeCaughtUp) {
  robotsim_core::StateTimeline timeline;
  const auto first = timeline.next(1000000000);
  const auto reset = timeline.next(1200000000, true);
  EXPECT_TRUE(reset.generation_changed);
  EXPECT_NE(first.generation, reset.generation);
  EXPECT_EQ(reset.sequence, 0u);
}

TEST(RollbackInhibit, BlocksOldOutputAndAcceptanceUntilGenerationReset) {
  using Clock = LatestCommandBuffer::Clock;
  robotsim_core::RollbackInhibit inhibit;
  robotsim_core::StateTimeline timeline;
  LatestCommandBuffer buffer("unitree_g1:test", 1, 0.2, std::chrono::milliseconds(50));
  const auto now = Clock::now();
  inhibit.observeTick(1000);
  const auto old_state = timeline.next(1000000000);
  buffer.resetGeneration(old_state.generation);

  robotsim_interfaces::msg::RobotCommand old_command;
  old_command.schema_version = "1.0";
  old_command.model_identity = "unitree_g1:test";
  old_command.generation = old_state.generation;
  old_command.source_id = "test";
  old_command.command_type = old_command.COMMAND_TORQUE;
  old_command.joint_names = {"joint"};
  old_command.effort_nm = {0.1};
  bool accepted = false;
  inhibit.synchronize([&](bool pending) {
    if (!pending) accepted = buffer.accept(old_command, old_state.generation, now);
  });
  ASSERT_TRUE(accepted);

  inhibit.observeTick(900);
  ASSERT_TRUE(inhibit.snapshot().pending);

  std::optional<robotsim_interfaces::msg::RobotCommand> emitted;
  inhibit.synchronize([&](bool pending) {
    if (!pending) emitted = buffer.current(now);
  });
  EXPECT_FALSE(emitted.has_value());

  old_command.sequence = 1;
  accepted = true;
  inhibit.synchronize([&](bool pending) {
    if (pending) {
      accepted = false;
      return;
    }
    accepted = buffer.accept(old_command, old_state.generation, now);
  });
  EXPECT_FALSE(accepted);

  const auto reset_snapshot = inhibit.snapshot();
  const auto new_state = timeline.next(900000000, reset_snapshot.pending);
  ASSERT_TRUE(new_state.generation_changed);
  buffer.resetGeneration(new_state.generation);
  inhibit.generationEstablished(reset_snapshot.epoch);
  EXPECT_FALSE(inhibit.snapshot().pending);
  EXPECT_FALSE(buffer.current(now).has_value());

  auto new_command = old_command;
  new_command.generation = new_state.generation;
  new_command.sequence = 0;
  accepted = false;
  inhibit.synchronize([&](bool pending) {
    if (!pending) accepted = buffer.accept(new_command, new_state.generation, now);
  });
  ASSERT_TRUE(accepted);
  inhibit.synchronize([&](bool pending) {
    if (!pending) emitted = buffer.current(now);
  });
  ASSERT_TRUE(emitted.has_value());
  EXPECT_EQ(emitted->generation, new_state.generation);
  EXPECT_EQ(emitted->sequence, 0u);
}

TEST(RollbackInhibit, KeepsNewerRollbackPendingWhenEarlierEpochIsCommitted) {
  robotsim_core::RollbackInhibit inhibit;
  inhibit.observeTick(1000);
  inhibit.observeTick(900);
  const auto first_rollback = inhibit.snapshot();
  inhibit.observeTick(0);
  inhibit.generationEstablished(first_rollback.epoch);
  EXPECT_TRUE(inhibit.snapshot().pending);
}

TEST(LatestCommandBuffer, ReplacesLatestAndExpiresBySteadyReceiptTime) {
  LatestCommandBuffer buffer("unitree_g1:test", 2, 0.2, std::chrono::milliseconds(50));
  buffer.resetGeneration("g1");
  robotsim_interfaces::msg::RobotCommand command;
  command.schema_version = "1.0";
  command.model_identity = "unitree_g1:test";
  command.generation = "g1";
  command.source_id = "test";
  command.command_type = command.COMMAND_TORQUE;
  command.joint_names = {"a", "b"};
  command.effort_nm = {0.1, -0.1};
  const auto now = LatestCommandBuffer::Clock::now();
  EXPECT_TRUE(buffer.accept(command, "g1", now));
  EXPECT_EQ(buffer.current(now)->sequence, 0u);
  command.sequence = 1;
  command.effort_nm = {0.15, -0.15};
  EXPECT_TRUE(buffer.accept(command, "g1", now));
  EXPECT_EQ(buffer.coalesced(), 1u);
  EXPECT_EQ(buffer.current(now)->sequence, 1u);
  EXPECT_FALSE(buffer.current(now + std::chrono::milliseconds(51)));
}

TEST(LatestCommandBuffer, RejectsOldGenerationDuplicateAndOutOfRangeEffort) {
  LatestCommandBuffer buffer("unitree_g1:test", 1, 0.2, std::chrono::milliseconds(50));
  buffer.resetGeneration("g1");
  robotsim_interfaces::msg::RobotCommand command;
  command.schema_version = "1.0";
  command.model_identity = "unitree_g1:test";
  command.generation = "old";
  command.source_id = "test";
  command.command_type = command.COMMAND_TORQUE;
  command.joint_names = {"joint"};
  command.effort_nm = {0.1};
  std::string reason;
  EXPECT_FALSE(buffer.accept(command, "g1", LatestCommandBuffer::Clock::now(), &reason));
  EXPECT_EQ(reason, "generation mismatch");
  command.generation = "g1";
  command.sequence = 1;
  EXPECT_FALSE(buffer.accept(command, "g1", LatestCommandBuffer::Clock::now()));
  command.sequence = 0;
  EXPECT_TRUE(buffer.accept(command, "g1", LatestCommandBuffer::Clock::now()));
  EXPECT_FALSE(buffer.accept(command, "g1", LatestCommandBuffer::Clock::now()));
  command.sequence = 1;
  command.effort_nm = {0.21};
  EXPECT_FALSE(buffer.accept(command, "g1", LatestCommandBuffer::Clock::now()));
  command.effort_nm = {0.1};
  command.source_id = "other";
  EXPECT_FALSE(buffer.accept(command, "g1", LatestCommandBuffer::Clock::now()));
}
