cmake_minimum_required(VERSION 3.8)
project(sagan_odometry)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sagan_interfaces REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(sagan_odometry  src/sagan_odometry.cpp)

ament_target_dependencies(sagan_odometry
  rclcpp
  tf2_ros
  tf2_geometry_msgs
  nav_msgs
  geometry_msgs
  sagan_interfaces
)

install(TARGETS sagan_odometry
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
