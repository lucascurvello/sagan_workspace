#ifndef SAGAN_KALMAN_FILTER_HPP_
#define SAGAN_KALMAN_FILTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>

// Define the size of our state vector. Easy to modify later.
constexpr int STATE_SIZE = 8;

// Define type aliases for clarity
using StateVector = Eigen::Matrix<double, STATE_SIZE, 1>;
using StateCovariance = Eigen::Matrix<double, STATE_SIZE, STATE_SIZE>;
using ProcessNoiseCovariance = Eigen::Matrix<double, STATE_SIZE, STATE_SIZE>;


class SaganKalmanFilter : public rclcpp::Node
{
public:
    SaganKalmanFilter();

private:
    // --- Core Kalman Filter Functions ---
    void predict();
    void update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);

    // --- ROS 2 Callbacks ---
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void reset_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                        std::shared_ptr<std_srvs::srv::Trigger::Response> response);

    // --- Helper Functions ---
    void publish_fused_odometry();
    void declare_parameters();

    // --- State and Covariance Matrices ---
    StateVector x_;               // State vector [x, y, vx, vy, ax, ay, theta, omega]'
    StateCovariance P_;           // State covariance matrix
    ProcessNoiseCovariance Q_;    // Process noise covariance matrix

    // --- ROS 2 Members ---
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr fused_odom_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
    rclcpp::TimerBase::SharedPtr predict_timer_;

    // --- Timing ---
    rclcpp::Time last_predict_time_;
    
    // --- Sensor Configuration ---
    // These vectors will store which states each sensor measures.
    // e.g., for odom: [true, true, true, true, false, false, true, true] means it measures x, y, vx, vy, theta, omega
    std::vector<bool> odom_measurement_map_;
    std::vector<bool> imu_measurement_map_;

    // Noise matrices for each sensor, dynamically sized based on measurement map
    Eigen::MatrixXd R_odom_;
    Eigen::MatrixXd R_imu_;

    // --- Parameters ---
    std::string odom_frame_id_;
    std::string base_frame_id_;
    double predict_frequency_;
};

#endif // SAGAN_KALMAN_FILTER_HPP_
