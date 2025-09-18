#ifndef SAGAN_KALMAN_FILTER_HPP_
#define SAGAN_KALMAN_FILTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <Eigen/Dense>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class SaganKalmanFilter : public rclcpp::Node
{
public:
    SaganKalmanFilter();

private:
    void predict(double dt);
    void update_odom(const Eigen::Matrix<double, 6, 1>& z);
    void update_imu(const Eigen::Matrix<double, 3, 1>& z);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void publish_fused_odometry();

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr fused_odom_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    Eigen::Matrix<double, 8, 1> x_;
    Eigen::Matrix<double, 8, 8> P_;
    Eigen::Matrix<double, 8, 8> Q_;
    Eigen::Matrix<double, 6, 6> R_odom_;
    Eigen::Matrix<double, 3, 3> R_imu_;

    rclcpp::Time last_time_;
};

#endif // SAGAN_KALMAN_FILTER_HPP_