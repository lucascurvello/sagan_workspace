#ifndef SAGAN_KALMAN_FILTER_HPP_
#define SAGAN_KALMAN_FILTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "std_srvs/srv/trigger.hpp" 

#include <Eigen/Dense>

class SaganKalmanFilter : public rclcpp::Node
{
public:
    SaganKalmanFilter();

private:
    // State vector: [x, y, vx, vy, ax, ay, theta, omega]^T
    Eigen::Matrix<double, 8, 1> x_;
    // State covariance matrix
    Eigen::Matrix<double, 8, 8> P_;
    // Process noise covariance
    Eigen::Matrix<double, 8, 8> Q_;
    // Measurement noise covariance for odometry
    Eigen::Matrix<double, 6, 6> R_odom_;
    // Measurement noise covariance for IMU
    Eigen::Matrix<double, 3, 3> R_imu_;

    rclcpp::Time last_time_;

    // Subscribers and Publisher
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr fused_odom_pub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;

    // Initialize transform broadcaster
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Callback functions
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void reset_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request, std::shared_ptr<std_srvs::srv::Trigger::Response> response);

    // Kalman Filter steps
    void predict(double dt);
    void update_odom(const Eigen::Matrix<double, 6, 1>& z);
    void update_imu(const Eigen::Matrix<double, 3, 1>& z);

    // Helper function
    void publish_fused_odometry();
};

#endif // SAGAN_KALMAN_FILTER_HPP_
