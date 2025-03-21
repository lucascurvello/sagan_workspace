#ifndef SAGAN_DRIVE_CONTROLLER__SAGAN_DRIVE_CONTROLLER_HPP_
#define SAGAN_DRIVE_CONTROLLER__SAGAN_DRIVE_CONTROLLER_HPP_

#include <memory>
#include <string>

#include "controller_interface/controller_interface.hpp"
#include "controller_interface/helpers.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"

#include "sagan_interfaces/msg/sagan_cmd.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace sagan_drive_controller
{
class SaganDriverController : public controller_interface::ControllerInterface
{
public: 
  SaganDriverController();
    
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;

  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  controller_interface::CallbackReturn on_init() override;

  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  void CommandInterfacesUpdate();
  
protected:
  std::vector<std::string> joint_names_;
  std::vector<std::string> wheel_joint_names_;
  std::vector<std::string> steering_joint_names_;

  std::vector<std::string> fl_wheel_joint;
  std::vector<std::string> fr_wheel_joint;
  std::vector<std::string> rl_wheel_joint;
  std::vector<std::string> rr_wheel_joint;

  std::vector<std::string> fl_steering_joint;
  std::vector<std::string> fr_steering_joint;
  std::vector<std::string> rl_steering_joint;
  std::vector<std::string> rr_steering_joint;


  std::vector<std::string> wheel_command_interface_types_;
  std::vector<std::string> steering_command_interface_types_;
  std::vector<std::string> command_interface_types_;
  std::vector<std::string> state_interface_types_;
  
  std::vector<double>  wheel_command_interface_;
  std::vector<double>  steering_command_interface_;


  const std::vector<std::string> allowed_state_interface_types_ = {
    hardware_interface::HW_IF_POSITION,
    hardware_interface::HW_IF_VELOCITY,
    hardware_interface::HW_IF_EFFORT,
  };

  const std::vector<std::string> allowed_command_interface_types_ = {
    hardware_interface::HW_IF_POSITION,
    hardware_interface::HW_IF_VELOCITY,
  };

  template <typename T>
  using InterfaceReferences = std::vector<std::vector<std::reference_wrapper<T>>>;

  InterfaceReferences<hardware_interface::LoanedCommandInterface> joint_command_interface_;
  InterfaceReferences<hardware_interface::LoanedStateInterface> joint_state_interface_;

  std::mutex mutex_actuator;
  rclcpp::Subscription<sagan_interfaces::msg::SaganCmd>::SharedPtr joints_reference_subscriber_;

};

} // namespace sagan_drive_controller

#endif // SAGAN_DRIVE_CONTROLLER_SAGAN_DRIVE_CONTROLLER_HPP_