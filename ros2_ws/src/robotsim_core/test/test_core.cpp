#include "robotsim_core/latest_command_buffer.hpp"
#include "robotsim_core/state_timeline.hpp"

#include <gtest/gtest.h>

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
