cmake_minimum_required(VERSION 3.8)
project(sagan_drive_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sagan_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(sagan_drive_controller SHARED
  src/sagan_drive_controller.cpp
)

target_include_directories(sagan_drive_controller
  PUBLIC
    include
    controller_interface
    hardware_interface
    pluginlib
    rclcpp
    rclcpp_lifecycle
    sagan_interfaces
    std_msgs
)

ament_target_dependencies(sagan_drive_controller PUBLIC
  controller_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  sagan_interfaces
  std_msgs
)


  # Export for pluginlib description file
pluginlib_export_plugin_description_file(controller_interface sagan_drive_controller.xml)

# Add install directives
install(TARGETS sagan_drive_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

ament_export_include_directories(
    include
   )

ament_export_dependencies(
    controller_interface
    pluginlib
    rclcpp
    rclcpp_lifecycle
   )

ament_export_libraries(sagan_drive_controller)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
