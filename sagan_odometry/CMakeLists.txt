cmake_minimum_required(VERSION 3.8)
project(sagan_odometry)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find all necessary ROS 2 and system dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sagan_interfaces REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# This command creates the executable named 'sagan_odometry'
# from the specified source file.
add_executable(sagan_odometry src/sagan_odometry.cpp)

# This links the necessary libraries to your executable.
ament_target_dependencies(sagan_odometry
  rclcpp
  sagan_interfaces
  nav_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
)

# MODERN CMAKE BEST PRACTICE: This correctly and robustly ensures that
# header files in the 'include' directory are found by the compiler for this
# specific target. The older include_directories() has been removed.
target_include_directories(sagan_odometry PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Install the executable so 'ros2 run' can find it.
install(TARGETS sagan_odometry
  DESTINATION lib/${PROJECT_NAME}
)

# Install the header files to the standard ROS 2 location.
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

# Add linters for testing if BUILD_TESTING is enabled.
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

