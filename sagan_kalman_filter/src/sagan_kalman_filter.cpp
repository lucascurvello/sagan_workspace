#include "sagan_kalman_filter/sagan_kalman_filter.hpp"
#include <memory>
#include <geometry_msgs/msg/transform_stamped.hpp>

SaganKalmanFilter::SaganKalmanFilter()
: Node("sagan_kalman_filter")
{
    // Declare parameters for the diagonal elements of the covariance matrices
    this->declare_parameter<std::vector<double>>("Q_diag", 
        {0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1000.0});
    
    this->declare_parameter<std::vector<double>>("R_odom_diag", 
        {10e9, 10e9, 100.0, 100.0, 10e9, 10e9});

    this->declare_parameter<std::vector<double>>("R_imu_diag", 
        {10e8, 10e8, 0.1});

    // Get the parameter values
    std::vector<double> q_diag = this->get_parameter("Q_diag").as_double_array();
    std::vector<double> r_odom_diag = this->get_parameter("R_odom_diag").as_double_array();
    std::vector<double> r_imu_diag = this->get_parameter("R_imu_diag").as_double_array();
    
    // --- Initialize state and covariance ---
    x_ = Eigen::Matrix<double, 8, 1>::Zero();
    P_ = Eigen::Matrix<double, 8, 8>::Identity() * 1000.0;

    // --- Process noise covariance (Q) ---
    // Validate size and construct the matrix from the parameter
    if (q_diag.size() != 8) {
        RCLCPP_ERROR(this->get_logger(), "Q_diag parameter must have 8 elements. Using defaults.");
        q_diag = {0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1000.0};
    }
    Q_ = Eigen::Matrix<double, 8, 8>::Zero();
    for(int i = 0; i < 8; ++i) {
        Q_(i, i) = q_diag[i];
    }

    // --- Odometry measurement noise covariance (R_odom) ---
    if (r_odom_diag.size() != 6) {
        RCLCPP_ERROR(this->get_logger(), "R_odom_diag parameter must have 6 elements. Using defaults.");
        r_odom_diag = {10e9, 10e9, 100.0, 100.0, 10e9, 10e9};
    }
    R_odom_ = Eigen::Matrix<double, 6, 6>::Zero();
    for(int i = 0; i < 6; ++i) {
        R_odom_(i, i) = r_odom_diag[i];
    }

    // --- IMU measurement noise covariance (R_imu) ---
    if (r_imu_diag.size() != 3) {
        RCLCPP_ERROR(this->get_logger(), "R_imu_diag parameter must have 3 elements. Using defaults.");
        r_imu_diag = {10e8, 10e8, 0.1};
    }
    R_imu_ = Eigen::Matrix<double, 3, 3>::Zero();
    for(int i = 0; i < 3; ++i) {
        R_imu_(i, i) = r_imu_diag[i];
    }

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

    reset_service_ = this->create_service<std_srvs::srv::Trigger>(
        "reset_ekf",
        std::bind(&SaganKalmanFilter::reset_callback, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Kalman Filter Node has been started.");
}

void SaganKalmanFilter::predict(double dt)
{
    // Extract current state variables
    double vx = x_(2);
    double vy = x_(3);
    double ax = x_(4);
    double ay = x_(5);
    double theta = x_(6);
    double omega = x_(7);
    double slipage_coefficient = -0.166508;

    // Pre-calculate sin and cos of theta
    double ct = std::cos(theta);
    double st = std::sin(theta);
    double dt2 = dt * dt;

    // --- UPDATED JACOBIAN (F) MATRIX CALCULATION ---
    Eigen::Matrix<double, 8, 8> F = Eigen::Matrix<double, 8, 8>::Identity();
    
    // Row 0 (x_next) - Unchanged
    F(0, 2) = ct * dt;
    F(0, 3) = -st * dt;
    F(0, 4) = 0.5 * ct * dt2;
    F(0, 5) = -0.5 * st * dt2;
    F(0, 6) = (-vx * st - vy * ct) * dt + 0.5 * (-ax * st - ay * ct) * dt2;
    
    // Row 1 (y_next) - Unchanged
    F(1, 2) = st * dt;
    F(1, 3) = ct * dt;
    F(1, 4) = 0.5 * st * dt2;
    F(1, 5) = 0.5 * ct * dt2;
    F(1, 6) = (vx * ct - vy * st) * dt + 0.5 * (ax * ct - ay * st) * dt2;

    // Row 2 (vx_next) - Unchanged
    F(2, 4) = dt;

    // Row 3 (vy_next) - THIS IS THE CORRECTED ROW
    F(3, 3) = 0.0; // Partial derivative of vy_next w.r.t vy is now 0
    F(3, 5) = 0.0; // Partial derivative of vy_next w.r.t ay is now 0
    F(3, 7) = slipage_coefficient; // Partial derivative of vy_next w.r.t omega

    // Row 6 (theta_next) - Unchanged
    F(6, 7) = dt;

    // --- Predict state using your new motion model ---
    Eigen::Matrix<double, 8, 1> x_pred = x_;
    x_pred(0) = x_(0) + (vx*ct - vy*st)*dt + 0.5*(ax*ct - ay*st)*dt2;
    x_pred(1) = x_(1) + (vx*st + vy*ct)*dt + 0.5*(ax*st + ay*ct)*dt2;
    x_pred(2) = vx + ax * dt;
    x_pred(3) = slipage_coefficient * omega; // Your new model for vy
    x_pred(4) = ax;
    x_pred(5) = ay;
    x_pred(6) = theta + omega * dt;
    x_pred(7) = omega;
    x_ = x_pred;
    
    // Predict covariance using the updated Jacobian
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
    double ay = msg->linear_acceleration.y;
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

void SaganKalmanFilter::reset_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    // The request is empty, so we don't use it.
    (void)request;

    // --- THIS IS YOUR RESET LOGIC ---
    RCLCPP_INFO(this->get_logger(), "Reset kalman filter command received!");
    x_ = Eigen::Matrix<double, 8, 1>::Zero();
    P_ = Eigen::Matrix<double, 8, 8>::Identity() * 1000.0;
    RCLCPP_INFO(this->get_logger(), "Kalman Filter RESETED");
    // --------------------------------

    // The response indicates if the command was successful
    response->success = true;
    response->message = "Kalman Filter successfully reset.";
}


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SaganKalmanFilter>());
    rclcpp::shutdown();
    return 0;
}

