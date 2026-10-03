#include "robotsim_g1_adapter/g1_joint_map.hpp"
#include "robotsim_g1_adapter/sim_time.hpp"

#include <gtest/gtest.h>

TEST(G1JointMap, ExplicitNamesReorderToSdkIndices) {
  std::vector<std::string> names;
  std::vector<double> values;
  for (std::size_t i = robotsim_g1_adapter::kG1JointNames.size(); i > 0; --i) {
    names.emplace_back(robotsim_g1_adapter::kG1JointNames[i - 1]);
    values.push_back(static_cast<double>(i));
  }
  const auto ordered = robotsim_g1_adapter::toSdkOrder(names, values);
  for (std::size_t i = 0; i < ordered.size(); ++i)
    EXPECT_DOUBLE_EQ(ordered[i], static_cast<double>(i + 1));
}

TEST(G1JointMap, RejectsUnknownDuplicateAndIncompleteNames) {
  std::vector<std::string> names;
  std::vector<double> values(29, 0.0);
  for (const auto* name : robotsim_g1_adapter::kG1JointNames) names.emplace_back(name);
  auto unknown = names;
  unknown[0] = "not_a_g1_joint";
  EXPECT_THROW(robotsim_g1_adapter::toSdkOrder(unknown, values), std::invalid_argument);
  auto duplicate = names;
  duplicate[1] = duplicate[0];
  EXPECT_THROW(robotsim_g1_adapter::toSdkOrder(duplicate, values), std::invalid_argument);
  names.pop_back();
  values.pop_back();
  EXPECT_THROW(robotsim_g1_adapter::toSdkOrder(names, values), std::invalid_argument);
}

TEST(G1SimTime, ConvertsBackendTickMillisecondsToRosTime) {
  const auto stamp = robotsim_g1_adapter::tickMillisecondsToRosTime(1234);
  EXPECT_EQ(stamp.sec, 1);
  EXPECT_EQ(stamp.nanosec, 234000000u);
  const auto zero = robotsim_g1_adapter::tickMillisecondsToRosTime(0);
  EXPECT_EQ(zero.sec, 0);
  EXPECT_EQ(zero.nanosec, 0u);
}
