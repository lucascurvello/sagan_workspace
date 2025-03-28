#ifndef sagan_drive_controller__SAGAN_DRIVE_CONTROLLER_CPP_
#define sagan_drive_controller__SAGAN_DRIVE_CONTROLLER_CPP_

#include <sagan_drive_controller/sagan_drive_controller.hpp>
#include <string>
#include <vector>

#include "lifecycle_msgs/msg/state.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/parameter.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace sagan_drive_controller
{

SaganDriverController::SaganDriverController()
  : controller_interface::ControllerInterface(),
    joint_names_({})
{
  fprintf(stderr, "Contruction done");
}

controller_interface::CallbackReturn SaganDriverController::on_init()
{
  try
  {
    //auto_declare<std::vector<std::string>>("joints", joint_names_);
    
    auto_declare<std::vector<std::string>>("command_interfaces", command_interface_types_);
    
    auto_declare<std::vector<std::string>>("wheel_command_interfaces", wheel_command_interface_types_);
    auto_declare<std::vector<std::string>>("steering_command_interfaces", steering_command_interface_types_);
    auto_declare<std::vector<std::string>>("state_interfaces", state_interface_types_);
    //auto_declare<std::vector<double>>("joints_references", {});

    auto_declare<std::vector<std::string>>("front_left_wheel_joint", fl_wheel_joint);
    auto_declare<std::vector<std::string>>("front_right_wheel_joint", fr_wheel_joint);
    auto_declare<std::vector<std::string>>("rear_left_wheel_joint", rl_wheel_joint);
    auto_declare<std::vector<std::string>>("rear_right_wheel_joint", rr_wheel_joint);

    auto_declare<std::vector<std::string>>("front_left_steering_joint", fl_steering_joint);
    auto_declare<std::vector<std::string>>("front_right_steering_joint", fr_steering_joint);
    auto_declare<std::vector<std::string>>("rear_left_steering_joint", rl_steering_joint);
    auto_declare<std::vector<std::string>>("rear_right_steering_joint", rr_steering_joint);
  }
  catch (const std::exception & e)
  {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn SaganDriverController::on_configure(
  const rclcpp_lifecycle::State &)    
{
  auto logger = get_node()->get_logger();

  fl_wheel_joint = get_node()->get_parameter("front_left_wheel_joint").as_string_array();
  fr_wheel_joint = get_node()->get_parameter("front_right_wheel_joint").as_string_array();
  rl_wheel_joint = get_node()->get_parameter("rear_left_wheel_joint").as_string_array();
  rr_wheel_joint = get_node()->get_parameter("rear_right_wheel_joint").as_string_array();

  fl_steering_joint = get_node()->get_parameter("front_left_steering_joint").as_string_array();
  fr_steering_joint = get_node()->get_parameter("front_right_steering_joint").as_string_array();
  rl_steering_joint = get_node()->get_parameter("rear_left_steering_joint").as_string_array();
  rr_steering_joint = get_node()->get_parameter("rear_right_steering_joint").as_string_array();

  
  if (wheel_command_interface_types_.empty())
  {
    wheel_command_interface_types_ = get_node()->get_parameter("wheel_command_interfaces").as_string_array();
  }

  if (wheel_command_interface_types_.empty())
  {
    RCLCPP_ERROR(logger, "'w_command_interfaces' parameter is empty.");
    return CallbackReturn::FAILURE;
  }

  if (steering_command_interface_types_.empty())
  {
    steering_command_interface_types_ = get_node()->get_parameter("steering_command_interfaces").as_string_array();
  }

  if (steering_command_interface_types_.empty())
  {
    RCLCPP_ERROR(logger, "'s_command_interfaces' parameter is empty.");
    return CallbackReturn::FAILURE;
  }

  
  if (command_interface_types_.empty())
  {
    command_interface_types_ = get_node()->get_parameter("command_interfaces").as_string_array();
  }

  if (command_interface_types_.empty())
  {
    RCLCPP_ERROR(logger, "'command_interfaces' parameter is empty.");
    return CallbackReturn::FAILURE;
  }
  
  

  for (const auto &interface : command_interface_types_)
  {
    auto it =
      std::find(allowed_command_interface_types_.begin(), allowed_command_interface_types_.end(), interface);
    if (it == allowed_command_interface_types_.end())
    {
      RCLCPP_ERROR(logger, "Command interface type '%s' not allowed! Only effort type is allowed!", interface.c_str());
      return CallbackReturn::FAILURE;
    }
  }

  joint_command_interface_.resize(allowed_command_interface_types_.size());

  // State interface checking
  state_interface_types_ = get_node()->get_parameter("state_interfaces").as_string_array();

  if (state_interface_types_.empty())
  {
  RCLCPP_ERROR(logger, "'state_interfaces' parameter is empty.");
  return CallbackReturn::FAILURE;
  }

  joint_state_interface_.resize(allowed_state_interface_types_.size());

  // Pritig format
  auto get_interface_list = [](const std::vector<std::string> &interface_types)
  {
    std::stringstream ss_interfaces;
    for (size_t index = 0; index < interface_types.size(); ++index)
    {
      if (index != 0)
     {
      ss_interfaces << " ";
      }
      ss_interfaces << interface_types[index];
    }
    return ss_interfaces.str();
  };

  RCLCPP_INFO(
    logger, "Command interfaces are [%s] and and state interfaces are [%s].",
    get_interface_list(command_interface_types_).c_str(),
    get_interface_list(state_interface_types_).c_str());

  
  joints_reference_subscriber_ = get_node()->create_subscription<sagan_interfaces::msg::SaganCmd>(
    "/SaganCommands", rclcpp::SystemDefaultsQoS(),
    [this](const std::shared_ptr<sagan_interfaces::msg::SaganCmd> msg) -> void
    {
        std::lock_guard<std::mutex> lock(this->mutex_actuator);
        for (int index = 0; index < 4; index++)
        {
          wheel_command_interface_[index] = msg->wheel_cmd[index].angular_velocity;
          steering_command_interface_[index] = msg->steering_cmd[index].angular_position;
        //     // qr[index] = msg->motor_cmd[index].q;
        //     // dqr[index] = msg->motor_cmd[index].dq;
        //     // kd[index] = msg->motor_cmd[index].kd;
        //     // kp[index] = msg->motor_cmd[index].kp;
        //     // tau[index] = msg->motor_cmd[index].tau;
            
        }
    });

    joints_state_publisher_ = get_node()->create_publisher<sagan_interfaces::msg::SaganStates>("Sagan/SaganStates", 10);

  return CallbackReturn::SUCCESS;

}

controller_interface::InterfaceConfiguration
SaganDriverController::command_interface_configuration() const
{

  controller_interface::InterfaceConfiguration command_interfaces_config;
  command_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  command_interfaces_config.names.reserve(8 * wheel_command_interface_types_.size());

  std::vector<std::string> wheel_joint_names_ = {fl_wheel_joint[0], fr_wheel_joint[0], rl_wheel_joint[0], rr_wheel_joint[0]};
  std::vector<std::string> steering_joint_names_ = {fl_steering_joint[0], fr_steering_joint[0], rl_steering_joint[0], rr_steering_joint[0]};

  for (std::vector<std::string>::iterator it = wheel_joint_names_.begin(); it != wheel_joint_names_.end(); ++it){
    for (const auto &interface_type : wheel_command_interface_types_)
    {
      command_interfaces_config.names.push_back(*it + "/" + interface_type);
    }
  }

  for (std::vector<std::string>::iterator it = steering_joint_names_.begin(); it != steering_joint_names_.end(); ++it){
    for (const auto &interface_type : steering_command_interface_types_)
    {
      command_interfaces_config.names.push_back(*it + "/" + interface_type);
    }
  }

  return command_interfaces_config;
}


controller_interface::InterfaceConfiguration
SaganDriverController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration state_interfaces_config;
  state_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  state_interfaces_config.names.reserve(8 * state_interface_types_.size());

  std::vector<std::string> wheel_joint_names_ = {fl_wheel_joint[0], fr_wheel_joint[0], rl_wheel_joint[0], rr_wheel_joint[0]};
  std::vector<std::string> steering_joint_names_ = {fl_steering_joint[0], fr_steering_joint[0], rl_steering_joint[0], rr_steering_joint[0]};

  for (std::vector<std::string>::iterator it = wheel_joint_names_.begin(); it != wheel_joint_names_.end(); ++it){
    for (const auto &interface_type : state_interface_types_)
    {
      state_interfaces_config.names.push_back(*it + "/" + interface_type);
    }
  }

  for (std::vector<std::string>::iterator it = steering_joint_names_.begin(); it != steering_joint_names_.end(); ++it){
    for (const auto &interface_type : state_interface_types_)
    {
      state_interfaces_config.names.push_back(*it + "/" + interface_type);
    }
  }

  return state_interfaces_config;
}

controller_interface::CallbackReturn SaganDriverController::on_activate(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_DEBUG(get_node()->get_logger(), "Activating");
  RCLCPP_INFO(get_node()->get_logger(), "Activating");

  std::vector<std::string> wheel_joint_names_ = {fl_wheel_joint[0], fr_wheel_joint[0], rl_wheel_joint[0], rr_wheel_joint[0]};
  std::vector<std::string> steering_joint_names_ = {fl_steering_joint[0], fr_steering_joint[0], rl_steering_joint[0], rr_steering_joint[0]};

  std::vector<std::string> joint_names_ = wheel_joint_names_;
  joint_names_.insert(joint_names_.end(),steering_joint_names_.begin(),steering_joint_names_.end());

  wheel_command_interface_.resize(wheel_joint_names_.size());
  steering_command_interface_.resize(steering_joint_names_.size());

  for (size_t i = 0; i < 4; i++)
  {
    wheel_command_interface_[i] = double(0.0);
    steering_command_interface_[i] = double(0.0);
  }
  
  
  // for (const auto &interface : command_interface_types_)
  // {
  //   auto it =
  //     std::find(allowed_command_interface_types_.begin(), allowed_command_interface_types_.end(), interface);
  //   auto index = std::distance(allowed_command_interface_types_.begin(), it);
  //   if (!controller_interface::get_ordered_interfaces(command_interfaces_, joint_names_, interface, joint_command_interface_[index]))
  //   {
  //     RCLCPP_ERROR(
  //       get_node()->get_logger(), "Expected %zu '%s' command interfaces, got %zu.", joint_names_.size(),
  //       interface.c_str(), joint_command_interface_[index].size());
  //     return CallbackReturn::ERROR;
  //   }
  // }

  for (const auto &interface : state_interface_types_)
  {
    auto it =
      std::find(allowed_state_interface_types_.begin(), allowed_state_interface_types_.end(), interface);
    auto index = std::distance(allowed_state_interface_types_.begin(), it);
    if (!controller_interface::get_ordered_interfaces(state_interfaces_, joint_names_, interface, joint_state_interface_[index]))
    {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Expected %zu '%s' state interfaces, got %zu.", joint_names_.size(),
        interface.c_str(), joint_state_interface_[index].size());
      return CallbackReturn::ERROR;
    }
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn SaganDriverController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_DEBUG(get_node()->get_logger(), "Deactivating");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type SaganDriverController::update(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  const auto logger = get_node()->get_logger();
  // update dynamic parameters


  //for (auto index = 0; index < 4; index++)
  //{ 
  //  wheel_command_interface_[index] = 10.0;
  //  steering_command_interface_[index] = 3.14;
  //}

  CommandInterfacesUpdate();
  StatesPublisher();

  return controller_interface::return_type::OK;
}

void SaganDriverController::CommandInterfacesUpdate(){

  auto logger = get_node()->get_logger();

  for (auto index = 0; index < 4; index++)
  {
    if (!command_interfaces_[index].set_value(wheel_command_interface_[index])) {
        RCLCPP_ERROR(logger, "Failed to set value for wheel_command_interface_[%d]", index);
    }

    if (!command_interfaces_[index + 4].set_value(steering_command_interface_[index])) {
        RCLCPP_ERROR(logger, "Failed to set value for steering_command_interface_[%d]", index);
    }
  }
}

void SaganDriverController::StatesPublisher(){
  std::lock_guard<std::mutex> lock(this->mutex_actuator);

  for (auto index = 0; index < 4; index++){
    SaganStates_msg.wheel_state[index].angular_velocity = wheel_command_interface_[index];
    SaganStates_msg.steering_state[index].angular_position = steering_command_interface_[index];
  }

  joints_state_publisher_->publish(SaganStates_msg);
}

} // namespace sagan_drive_controller

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
  sagan_drive_controller::SaganDriverController, 
  controller_interface::ControllerInterface)
  
#endif //sagan_drive_controller_SAGAN_DRIVE_CONTROLLER_CPP_