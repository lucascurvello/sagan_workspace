#ifndef SAGAN_ODOMETRY_NODE_HPP_
#define SAGAN_ODOMETRY_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sagan_interfaces/msg/sagan_states.hpp>
#include "std_srvs/srv/trigger.hpp" 
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <random>
#include <string>
#include <chrono>

// The incorrect forward declaration that caused the error has been removed.
// Including the full message header above is the correct approach.

class SaganOdometryNode : public rclcpp::Node
{
public:
    /**
     * @brief Construct a new Sagan Odometry Node object
     */
    SaganOdometryNode();

private:
    /**
     * @brief Callback function for the /Sagan/SaganStates topic.
     * Stores clean and noisy wheel velocities.
     * @param msg The incoming SaganStates message.
     */
    void state_callback(const sagan_interfaces::msg::SaganStates::SharedPtr msg);

    /**
     * @brief Timer callback function to calculate and publish odometry.
     * Runs at a fixed frequency.
     */
    void timer_callback();

    void reset_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                        std::shared_ptr<std_srvs::srv::Trigger::Response> response);

    // --- ROS 2 Member Variables ---
    rclcpp::Subscription<sagan_interfaces::msg::SaganStates>::SharedPtr subscription_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_with_noise_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;

    // --- Robot State and Pose Variables ---
    // Clean (ground-truth) pose
    double x_, y_, theta_;
    // Noisy (simulated) pose
    double noisy_x_, noisy_y_, noisy_theta_;
    
    // --- Wheel State Variables ---
    double omega_clean_[4] = {0.0};
    double omega_noisy_[4] = {0.0};
    double delta_[4] = {0.0};

    // --- Noise Generation ---
    double wheel_velocity_noise_scaler_;
    std::default_random_engine random_generator_;
    std::normal_distribution<double> noise_distribution_;

    // --- Time ---
    rclcpp::Time last_time_;
};

#endif // SAGAN_ODOMETRY_NODE_HPP_

