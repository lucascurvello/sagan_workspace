#include "sagan_kalman_filter/sagan_kalman_filter.hpp"
#include <memory>
#include <geometry_msgs/msg/transform_stamped.hpp>

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
          0, 0, 0.1, 0, 0, 0, 0, 0,
          0, 0, 0, 0.001, 0, 0, 0, 0,
          0, 0, 0, 0, 1, 0, 0, 0,
          0, 0, 0, 0, 0, 1, 0, 0,
          0, 0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 1;

    // Measurement noise covariance
    R_odom_ = Eigen::Matrix<double, 6, 6>::Identity();
    R_odom_ << 10e9, 0, 0, 0, 0, 0,
               0, 10e9, 0, 0, 0, 0,
               0, 0, 100, 0, 0, 0,
               0, 0, 0, 100, 0, 0,
               0, 0, 0, 0, 10e9, 0,
               0, 0, 0, 0, 0, 100;

    R_imu_ = Eigen::Matrix<double, 3, 3>::Identity();
    R_imu_ << 0.1, 0, 0,
              0, 10e9, 0,
              0, 0, 0.1;

    last_time_ = this->get_clock()->now();

    // Subscribers
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom/with_noise", 10, std::bind(&SaganKalmanFilter::odom_callback, this, std::placeholders::_1));
    
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
    x_pred(0) = x_(0) + (vx*cos(theta) - vy*sin(theta))*dt + 0.5*(ax*cos(theta) - ay*sin(theta))*dt2;
    x_pred(1) = x_(1) + (vx*sin(theta) + vy*cos(theta))*dt + 0.5*(ax*sin(theta) + ay*cos(theta))*dt2;
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
    Eigen::Matrix<double, 6, 8> H = Eigen::Matrix<double, 6, 8>::Zero();
    H(0, 0) = 1; // x from x
    H(1, 1) = 1; // y from y
    H(2, 2) = 1; // vx from vx
    H(3, 3) = 1; // vy from vy
    H(4, 6) = 1; // yaw from theta
    H(5, 7) = 1; // omega from omega

    Eigen::Matrix<double, 6, 1> y = z - H * x_;
    Eigen::Matrix<double, 6, 6> S = H * P_ * H.transpose() + R_odom_;
    Eigen::Matrix<double, 8, 6> K = P_ * H.transpose() * S.inverse();

    x_ = x_ + K * y;
    P_ = (Eigen::Matrix<double, 8, 8>::Identity() - K * H) * P_;
}

void SaganKalmanFilter::update_imu(const Eigen::Matrix<double, 3, 1>& z)
{
    // Measurement matrix H for IMU
    Eigen::Matrix<double, 3, 8> H = Eigen::Matrix<double, 3, 8>::Zero();
    H(0, 4) = 1; // ax from ax
    H(1, 5) = 1; // ay from ay
    H(2, 7) = 1; // omega from omega

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
    if (dt <= 0.0) { // Also handle dt == 0
        RCLCPP_WARN_ONCE(this->get_logger(), "dt is zero or negative, skipping prediction.");
        last_time_ = current_time;
        return;
    }
    last_time_ = current_time;

    predict(dt);

    // Extract measurements from odometry
    double x_pos = msg->pose.pose.position.x;
    double y_pos = msg->pose.pose.position.y;
    double vx = msg->twist.twist.linear.x;
    double vy = msg->twist.twist.linear.y;
    double omega_twist = msg->twist.twist.angular.z;
    tf2::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    Eigen::Matrix<double, 6, 1> z_odom;
    z_odom << x_pos, y_pos, vx, vy, yaw, omega_twist;
    
    update_odom(z_odom);

    publish_fused_odometry();
}

void SaganKalmanFilter::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    double ax = -msg->linear_acceleration.x;
    double ay = -msg->linear_acceleration.y;
    double omega_imu = msg->angular_velocity.z;

    Eigen::Matrix<double, 3, 1> z_imu;
    z_imu << ax, ay, omega_imu;

    update_imu(z_imu);
}

// ======================================================================================
// === THIS IS THE CORRECTED AND COMPLETED FUNCTION =====================================
// ======================================================================================
void SaganKalmanFilter::publish_fused_odometry()
{
    auto msg = nav_msgs::msg::Odometry();
    msg.header.stamp = this->get_clock()->now();
    msg.header.frame_id = "odom";
    msg.child_frame_id = "efk_reference"; // Changed from base_link to match your usage

    // --- POSE ---
    // Set the position from our state vector
    msg.pose.pose.position.x = x_(0); // state 'x'
    msg.pose.pose.position.y = x_(1); // state 'y'
    msg.pose.pose.position.z = 0.0;   // Assuming a 2D robot

    // Set the orientation from our state vector
    tf2::Quaternion q;
    q.setRPY(0, 0, x_(6)); // roll, pitch, yaw (from state 'theta')
    msg.pose.pose.orientation = tf2::toMsg(q);

    // --- POSE COVARIANCE ---
    // The pose covariance is a 6x6 matrix in row-major order.
    // The order of variables is [x, y, z, roll, pitch, yaw]
    // We only fill in the parts relevant to our 2D state [x, y, yaw]
    for (int i=0; i<36; i++) {
        msg.pose.covariance[i] = 0;
    }
    // Mapping P_ indices to the 6x6 covariance matrix
    // P_ state map: 0->x, 1->y, 6->theta(yaw)
    msg.pose.covariance[0*6 + 0] = P_(0,0); // Var(x)
    msg.pose.covariance[0*6 + 1] = P_(0,1); // Cov(x,y)
    msg.pose.covariance[0*6 + 5] = P_(0,6); // Cov(x,yaw)
    
    msg.pose.covariance[1*6 + 0] = P_(1,0); // Cov(y,x)
    msg.pose.covariance[1*6 + 1] = P_(1,1); // Var(y)
    msg.pose.covariance[1*6 + 5] = P_(1,6); // Cov(y,yaw)

    msg.pose.covariance[5*6 + 0] = P_(6,0); // Cov(yaw,x)
    msg.pose.covariance[5*6 + 1] = P_(6,1); // Cov(yaw,y)
    msg.pose.covariance[5*6 + 5] = P_(6,6); // Var(yaw)

    // --- TWIST ---
    // Twist is in the child_frame_id (body frame). Our state vx, vy are also in body frame.
    msg.twist.twist.linear.x = x_(2);  // state 'vx'
    msg.twist.twist.linear.y = x_(3);  // state 'vy'
    msg.twist.twist.angular.z = x_(7); // state 'omega'

    // --- TWIST COVARIANCE ---
    // The twist covariance is a 6x6 matrix in row-major order.
    // The order is [vx, vy, vz, v_roll, v_pitch, v_yaw(omega)]
    // We only fill in the parts relevant to our state [vx, vy, omega]
    for (int i=0; i<36; i++) {
        msg.twist.covariance[i] = 0;
    }
    // Mapping P_ indices to the 6x6 covariance matrix
    // P_ state map: 2->vx, 3->vy, 7->omega
    msg.twist.covariance[0*6 + 0] = P_(2,2); // Var(vx)
    msg.twist.covariance[0*6 + 1] = P_(2,3); // Cov(vx,vy)
    msg.twist.covariance[0*6 + 5] = P_(2,7); // Cov(vx,omega)

    msg.twist.covariance[1*6 + 0] = P_(3,2); // Cov(vy,vx)
    msg.twist.covariance[1*6 + 1] = P_(3,3); // Var(vy)
    msg.twist.covariance[1*6 + 5] = P_(3,7); // Cov(vy,omega)

    msg.twist.covariance[5*6 + 0] = P_(7,2); // Cov(omega,vx)
    msg.twist.covariance[5*6 + 1] = P_(7,3); // Cov(omega,vy)
    msg.twist.covariance[5*6 + 5] = P_(7,7); // Var(omega)

    fused_odom_pub_->publish(msg);

    // --- BROADCAST TRANSFORM ---
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped.header.stamp = this->get_clock()->now();
    transform_stamped.header.frame_id = "odom";
    transform_stamped.child_frame_id = msg.child_frame_id; // Use the same child_frame_id
    transform_stamped.transform.translation.x = x_(0);
    transform_stamped.transform.translation.y = x_(1);
    transform_stamped.transform.translation.z = 0.0;
    transform_stamped.transform.rotation = tf2::toMsg(q);
    
    tf_broadcaster_->sendTransform(transform_stamped);
}


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SaganKalmanFilter>());
    rclcpp::shutdown();
    return 0;
}

