#include "sagan_kalman_filter/sagan_kalman_filter.hpp"
#include <memory>

SaganKalmanFilter::SaganKalmanFilter()
: Node("sagan_kalman_filter")
{
    // Initialize state and covariance
    x_ = Eigen::Matrix<double, 8, 1>::Zero();
    P_ = Eigen::Matrix<double, 8, 8>::Identity() * 1000.0;

    // Process noise covariance
    Q_ = Eigen::Matrix<double, 8, 8>::Identity();
    Q_ << 0.01, 0, 0, 0, 0, 0, 0, 0,
          0, 0.01, 0, 0, 0, 0, 0, 0,
          0, 0, 0.01, 0, 0, 0, 0, 0,
          0, 0, 0, 0.10, 0, 0, 0, 0,
          0, 0, 0, 0, 0.10, 0, 0, 0,
          0, 0, 0, 0, 0, 0.10, 0, 0,
          0, 0, 0, 0, 0, 0, 0.10, 0,
          0, 0, 0, 0, 0, 0, 0, 0.10;

    // Measurement noise covariance
    R_odom_ = Eigen::Matrix<double, 6, 6>::Identity();
    R_odom_ << 0.1, 0, 0, 0, 0, 0,
               0, 0.1, 0, 0, 0, 0,
               0, 0, 0.1, 0, 0, 0,
               0, 0, 0, 0.1, 0, 0,
               0, 0, 0, 0, 0.1, 0,
               0, 0, 0, 0, 0, 0.1;

    R_imu_ = Eigen::Matrix<double, 3, 3>::Identity();
    R_imu_ << 0.1, 0, 0,
              0, 0.1, 0,
              0, 0, 0.1;

    last_time_ = this->get_clock()->now();

    // Subscribers
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10, std::bind(&SaganKalmanFilter::odom_callback, this, std::placeholders::_1));
    
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu", 10, std::bind(&SaganKalmanFilter::imu_callback, this, std::placeholders::_1));

    // Publisher
    fused_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom/filtered", 10);

    // Initialize transform broadcaster
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    RCLCPP_INFO(this->get_logger(), "Kalman Filter Node has been started.");
}

void SaganKalmanFilter::predict(double dt)
{
    double vx = x_(2);
    double vy = x_(3);
    double ax = x_(4);
    double ay = x_(5);
    double theta = x_(6);
    double omega = x_(7);

    // Pre-calculate sin and cos of theta
    double ct = std::cos(theta);
    double st = std::sin(theta);
    double dt2 = dt * dt;

    // State transition matrix F
    Eigen::Matrix<double, 8, 8> F = Eigen::Matrix<double, 8, 8>::Identity();
    F(0, 2) = cos(theta) * dt;
    F(0, 3) = -sin(theta) * dt;
    F(0, 4) = 0.5 * ct * dt2;
    F(0, 5) = -0.5 * st * dt2;
    F(0, 6) = (-vx * st - vy * ct) * dt + 0.5 * (-ax * st - ay * ct) * dt2;
    F(1, 2) = st * dt;
    F(1, 3) = ct * dt;
    F(1, 4) = 0.5 * st * dt2;
    F(1, 5) = 0.5 * ct * dt2;
    F(1, 6) = (vx * ct - vy * st) * dt + 0.5 * (ax * ct - ay * st) * dt2;
    F(2, 4) = dt;
    F(3, 5) = dt;
    F(6, 7) = dt;

    // Predict state
    Eigen::Matrix<double, 8, 1> x_pred = x_;
    x_pred(0) = x_(0) + (vx*cos(theta) - vy*sin(theta))*dt + 0.5*(ax*cos(theta) - ay*sin(theta))*dt^2;
    x_pred(1) = x_(1) + (vx*sin(theta) + vy*cos(theta))*dt + 0.5*(ax*sin(theta) + ay*cos(theta))*dt^2;
    x_pred(2) = vx + ax * dt;
    x_pred(3) = vy + ay * dt;
    x_pred(4) = ax;
    x_pred(5) = ay;
    x_pred(6) = theta + omega * dt;
    x_pred(7) = omega;
    x_ = x_pred;
    
    // Predict covariance
    P_ = F * P_ * F.transpose() + Q_;
}

void SaganKalmanFilter::update_odom(const Eigen::Matrix<double, 6, 1>& z)
{
    // Measurement matrix H for odometry
    Eigen::Matrix<double, 6, 8> H;
    H << 1, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 1;

    Eigen::Matrix<double, 6, 1> y = z - H * x_;
    Eigen::Matrix<double, 6, 6> S = H * P_ * H.transpose() + R_odom_;
    Eigen::Matrix<double, 8, 6> K = P_ * H.transpose() * S.inverse();

    x_ = x_ + K * y;
    P_ = (Eigen::Matrix<double, 8, 8>::Identity() - K * H) * P_;
}

void SaganKalmanFilter::update_imu(const Eigen::Matrix<double, 3, 1>& z)
{
    // Measurement matrix H for IMU
    Eigen::Matrix<double, 3, 8> H;
    H << 0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 1;

    Eigen::Matrix<double, 3, 1> y = z - H * x_;
    Eigen::Matrix<double, 3, 3> S = H * P_ * H.transpose() + R_imu_;
    Eigen::Matrix<double, 8, 3> K = P_ * H.transpose() * S.inverse();

    x_ = x_ + K * y;
    P_ = (Eigen::Matrix<double, 8, 8>::Identity() - K * H) * P_;
}

void SaganKalmanFilter::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    rclcpp::Time current_time = this->get_clock()->now();
    double dt = (current_time - last_time_).seconds();
    if (dt < 0.0) {
        RCLCPP_WARN(this->get_logger(), "Time went backwards, resetting.");
        last_time_ = current_time;
        return;
    }
    last_time_ = current_time;

    predict(dt);

    // Extract measurements from odometry
    double x_pos = msg->pose.pose.position.x;
    double y_pos = msg->pose.pose.position.y;
    tf2::Quaternion q(
    msg->pose.pose.orientation.x,
    msg->pose.pose.orientation.y,
    msg->pose.pose.orientation.z,
    msg->pose.pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    Eigen::Matrix<double, 3, 1> z_odom;
    z_odom << x_pos, y_pos, yaw;
    
    update_odom(z_odom);
    
    // Also update linear velocity from odometry twist
    x_(3) = msg->twist.twist.linear.x;

    publish_fused_odometry();
}

void SaganKalmanFilter::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    Eigen::Matrix<double, 1, 1> z_imu;
    z_imu(0) = msg->angular_velocity.z;

    update_imu(z_imu);
}

void SaganKalmanFilter::publish_fused_odometry()
{
    
    
    auto msg = nav_msgs::msg::Odometry();
    msg.header.stamp = this->get_clock()->now();
    msg.header.frame_id = "odom";
    msg.child_frame_id = "efk_reference";

    msg.pose.pose.position.x = x_(0);
    msg.pose.pose.position.y = x_(1);
    msg.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0, 0, x_(2));
    msg.pose.pose.orientation = tf2::toMsg(q);

    for (int i=0; i<36; i++) {
        msg.pose.covariance[i] = 0;
    }

    msg.pose.covariance[0*6+0] = P_(0,0);
    msg.pose.covariance[0*6+1] = P_(0,1);
    msg.pose.covariance[1*6+0] = P_(1,0);
    msg.pose.covariance[1*6+1] = P_(1,1);
    msg.pose.covariance[5*6+5] = P_(2,2);
    
    msg.twist.twist.linear.x = x_(3);
    msg.twist.twist.angular.z = x_(4);

    for (int i=0; i<36; i++) {
        msg.twist.covariance[i] = 0;
    }
    msg.twist.covariance[0*6+0] = P_(3,3);
    msg.twist.covariance[5*6+5] = P_(4,4);

    // Broadcast Transform from odom to base_link
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped.header.stamp = this->get_clock()->now();
    transform_stamped.header.frame_id = "odom";
    transform_stamped.child_frame_id = "efk_reference";
    transform_stamped.transform.translation.x = x_(0);
    transform_stamped.transform.translation.y = x_(1);
    transform_stamped.transform.translation.z = 0.0;
    transform_stamped.transform.rotation= tf2::toMsg(q);

    fused_odom_pub_->publish(msg);
    tf_broadcaster_->sendTransform(transform_stamped);
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SaganKalmanFilter>());
    rclcpp::shutdown();
    return 0;
}
