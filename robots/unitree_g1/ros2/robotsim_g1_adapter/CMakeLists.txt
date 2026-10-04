cmake_minimum_required(VERSION 3.8)
project(robotsim_g1_adapter)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(robotsim_core REQUIRED)
find_package(robotsim_interfaces REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(unitree_sdk2 CONFIG REQUIRED)

add_executable(g1_ros2_adapter src/g1_ros2_adapter.cpp)
target_compile_features(g1_ros2_adapter PUBLIC cxx_std_17)
target_include_directories(g1_ros2_adapter PRIVATE include)
target_link_libraries(g1_ros2_adapter unitree_sdk2)
ament_target_dependencies(g1_ros2_adapter
  builtin_interfaces geometry_msgs rclcpp robotsim_core robotsim_interfaces
  rosgraph_msgs sensor_msgs tf2_ros tf2_msgs)

install(TARGETS g1_ros2_adapter DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_g1_joint_map test/test_g1_joint_map.cpp)
  target_include_directories(test_g1_joint_map PRIVATE include)
  ament_target_dependencies(test_g1_joint_map builtin_interfaces)
endif()

ament_package()
