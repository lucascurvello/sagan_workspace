#include "sagan_kalman_filter/sagan_kalman_filter.hpp"
#include <memory>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/parameter.hpp>
#include <cmath> // Required for M_PI

// Helper function to configure a sensor from parameters
void configure_sensor(
    rclcpp::Node* node, 
    const std::string& sensor_name,
    std::vector<bool>& measurement_map, 
    Eigen::MatrixXd& R_matrix)
{
    // Declare parameters for the sensor
    node->declare_parameter<std::vector<bool>>(sensor_name + ".measurement_map", std::vector<bool>(STATE_SIZE, false));
    node->declare_parameter<std::vector<double>>(sensor_name + ".R_diag", {});

    // Get parameters
    measurement_map = node->get_parameter(sensor_name + ".measurement_map").as_bool_array();
    auto r_diag = node->get_parameter(sensor_name + ".R_diag").as_double_array();

    if (measurement_map.size() != STATE_SIZE) {
        RCLCPP_FATAL(node->get_logger(), "Parameter '%s.measurement_map' must have %d elements!", sensor_name.c_str(), STATE_SIZE);
        throw std::runtime_error("Invalid measurement_map size for " + sensor_name);
    }

    int measurement_count = 0;
    for(bool enabled : measurement_map) {
        if(enabled) measurement_count++;
    }

    if (r_diag.size() != measurement_count) {
        RCLCPP_FATAL(node->get_logger(), "Parameter '%s.R_diag' must have %d elements (matching enabled measurements), but has %zu.", sensor_name.c_str(), measurement_count, r_diag.size());
        throw std::runtime_error("Invalid R_diag size for " + sensor_name);
    }
    
    // Construct the R matrix
    R_matrix = Eigen::MatrixXd::Zero(measurement_count, measurement_count);
    int diag_index = 0;
    for (double val : r_diag) {
        R_matrix(diag_index, diag_index) = val;
        diag_index++;
    }
    RCLCPP_INFO(node->get_logger(), "Configured sensor '%s' with %d active measurements.", sensor_name.c_str(), measurement_count);
}


SaganKalmanFilter::SaganKalmanFilter()
: Node("sagan_kalman_filter")
{
    // --- Declare general parameters ---
    this->declare_parameter<std::vector<double>>("Q_diag", std::vector<double>(STATE_SIZE, 0.1));
    this->declare_parameter<std::string>("odom_frame_id", "odom");
    this->declare_parameter<std::string>("base_frame_id", "base_link");
    this->declare_parameter<double>("predict_frequency", 100.0);

    // --- Get general parameters ---
    odom_frame_id_ = this->get_parameter("odom_frame_id").as_string();
    base_frame_id_ = this->get_parameter("base_frame_id").as_string();
    predict_frequency_ = this->get_parameter("predict_frequency").as_double();
    std::vector<double> q_diag = this->get_parameter("Q_diag").as_double_array();

    // --- Initialize state and covariance ---
    x_ = StateVector::Zero();
    P_ = StateCovariance::Identity() * 1000.0; // High initial uncertainty

    // --- Process noise covariance (Q) ---
    if (q_diag.size() != STATE_SIZE) {
        RCLCPP_FATAL(this->get_logger(), "Q_diag parameter must have %d elements. Has %zu.", STATE_SIZE, q_diag.size());
        throw std::runtime_error("Invalid Q_diag size.");
    }
    Q_ = ProcessNoiseCovariance::Zero();
    for(int i = 0; i < STATE_SIZE; ++i) {
        Q_(i, i) = q_diag[i];
    }
    
    // --- Configure Sensors from Parameters ---
    try {
        configure_sensor(this, "odom_sensor", odom_measurement_map_, R_odom_);
        configure_sensor(this, "imu_sensor", imu_measurement_map_, R_imu_);
    } catch (const std::runtime_error& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to configure sensors: %s. Shutting down.", e.what());
        rclcpp::shutdown();
        return;
    }

    // --- ROS 2 Communications ---
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom/with_noise", 10, std::bind(&SaganKalmanFilter::odom_callback, this, std::placeholders::_1));
    
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu", 10, std::bind(&SaganKalmanFilter::imu_callback, this, std::placeholders::_1));

    fused_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom/filtered", 10);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    reset_service_ = this->create_service<std_srvs::srv::Trigger>(
        "reset_ekf",
        std::bind(&SaganKalmanFilter::reset_callback, this, std::placeholders::_1, std::placeholders::_2));
    
    // --- Prediction Timer ---
    last_predict_time_ = this->get_clock()->now();
    auto predict_period = std::chrono::duration<double>(1.0 / predict_frequency_);
    predict_timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(predict_period),
        std::bind(&SaganKalmanFilter::predict, this));

    RCLCPP_INFO(this->get_logger(), "Kalman Filter Node has been started.");
}

void SaganKalmanFilter::predict()
{
    rclcpp::Time current_time = this->get_clock()->now();
    double dt = (current_time - last_predict_time_).seconds();
    if (dt <= 0.0) {
        return;
    }
    last_predict_time_ = current_time;

    // Extract current state variables
    double vx = x_(2);
    double vy = x_(3);
    double ax = x_(4);
    double ay = x_(5);
    double theta = x_(6);
    double omega = x_(7);
    double slipage_coefficient = -0.166508; // This could also be a parameter

    // Pre-calculate sin and cos of theta
    double ct = std::cos(theta);
    double st = std::sin(theta);
    double dt2 = dt * dt;

    // --- Jacobian (F) Matrix Calculation ---
    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> F = Eigen::Matrix<double, STATE_SIZE, STATE_SIZE>::Identity();
    
    F(0, 2) = ct * dt; F(0, 3) = -st * dt; F(0, 4) = 0.5 * ct * dt2; F(0, 5) = -0.5 * st * dt2;
    F(0, 6) = (-vx * st - vy * ct) * dt + 0.5 * (-ax * st - ay * ct) * dt2;
    
    F(1, 2) = st * dt; F(1, 3) = ct * dt; F(1, 4) = 0.5 * st * dt2; F(1, 5) = 0.5 * ct * dt2;
    F(1, 6) = (vx * ct - vy * st) * dt + 0.5 * (ax * ct - ay * st) * dt2;

    F(2, 4) = dt;

    F(3, 3) = 0.0; F(3, 5) = 0.0; F(3, 7) = slipage_coefficient;

    F(6, 7) = dt;

    // --- Predict state using motion model ---
    StateVector x_pred = x_;
    x_pred(0) = x_(0) + (vx*ct - vy*st)*dt + 0.5*(ax*ct - ay*st)*dt2;
    x_pred(1) = x_(1) + (vx*st + vy*ct)*dt + 0.5*(ax*st + ay*ct)*dt2;
    x_pred(2) = vx + ax * dt;
    x_pred(3) = slipage_coefficient * omega;
    x_pred(4) = ax;
    x_pred(5) = ay;
    x_pred(6) = theta + omega * dt;
    x_pred(7) = omega;
    x_ = x_pred;
    
    // --- Predict covariance ---
    P_ = F * P_ * F.transpose() + Q_;
}

void SaganKalmanFilter::update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R)
{
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);

    Eigen::VectorXd y = z - H * x_; // Innovation or measurement residual

    // *** ANGLE WRAPPING CORRECTION ***
    // This is the critical fix. We check if any measurement corresponds to the
    // angle state (theta, index 6) and normalize its innovation (error).
    for (int i = 0; i < H.rows(); ++i) {
        if (H(i, 6) == 1.0) { 
            while (y(i) > M_PI) y(i) -= 2.0 * M_PI;
            while (y(i) < -M_PI) y(i) += 2.0 * M_PI;
        }
    }

    Eigen::MatrixXd S = H * P_ * H.transpose() + R; // Innovation covariance
    Eigen::MatrixXd K = P_ * H.transpose() * S.inverse(); // Kalman gain

    x_ = x_ + K * y;
    P_ = (I - K * H) * P_;
}

void SaganKalmanFilter::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    // 1. Collect all possible measurements from the message
    tf2::Quaternion q;
    tf2::fromMsg(msg->pose.pose.orientation, q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    // Full potential measurement vector from odometry
    double measurements[STATE_SIZE] = {
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        msg->twist.twist.linear.x,
        msg->twist.twist.linear.y,
        0.0, // Odometry typically doesn't measure linear acceleration
        0.0,
        yaw,
        msg->twist.twist.angular.z
    };

    // 2. Build z, H dynamically based on the odom_measurement_map_
    int active_measurements = R_odom_.rows();
    if (active_measurements == 0) return;

    Eigen::VectorXd z(active_measurements);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(active_measurements, STATE_SIZE);
    
    int current_row = 0;
    for (int i = 0; i < STATE_SIZE; ++i) {
        if (odom_measurement_map_[i]) {
            z(current_row) = measurements[i];
            H(current_row, i) = 1.0;
            current_row++;
        }
    }

    // 3. Call the generic update function
    update(z, H, R_odom_);

    // 4. Publish the result
    publish_fused_odometry();
}

void SaganKalmanFilter::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    // 1. Collect all possible measurements from the message
    tf2::Quaternion q;
    tf2::fromMsg(msg->orientation, q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    // Full potential measurement vector from IMU
    // Note: IMU x-acceleration is often forward, which corresponds to robot's -ax in some frames. Adjust if needed.
    double measurements[STATE_SIZE] = {
        0.0, // IMU does not measure position
        0.0,
        0.0, // IMU does not measure linear velocity
        0.0,
        -msg->linear_acceleration.x, // Mapping to our state `ax`
        msg->linear_acceleration.y,  // Mapping to our state `ay`
        yaw, // Some IMUs provide orientation
        msg->angular_velocity.z
    };
    
    // 2. Build z, H dynamically based on the imu_measurement_map_
    int active_measurements = R_imu_.rows();
    if (active_measurements == 0) return;

    Eigen::VectorXd z(active_measurements);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(active_measurements, STATE_SIZE);

    int current_row = 0;
    for (int i = 0; i < STATE_SIZE; ++i) {
        if (imu_measurement_map_[i]) {
            z(current_row) = measurements[i];
            H(current_row, i) = 1.0;
            current_row++;
        }
    }
    
    // 3. Call the generic update function
    update(z, H, R_imu_);
}

void SaganKalmanFilter::publish_fused_odometry()
{
    auto msg = nav_msgs::msg::Odometry();
    msg.header.stamp = this->get_clock()->now();
    msg.header.frame_id = odom_frame_id_;
    msg.child_frame_id = base_frame_id_;

    msg.pose.pose.position.x = x_(0);
    msg.pose.pose.position.y = x_(1);
    msg.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0, 0, x_(6));
    msg.pose.pose.orientation = tf2::toMsg(q);

    // Simplified covariance mapping for clarity.
    // Maps [x, y, yaw] from P_ to 6x6 pose covariance
    msg.pose.covariance.fill(0.0);
    msg.pose.covariance[0*6 + 0] = P_(0,0); msg.pose.covariance[0*6 + 1] = P_(0,1); msg.pose.covariance[0*6 + 5] = P_(0,6);
    msg.pose.covariance[1*6 + 0] = P_(1,0); msg.pose.covariance[1*6 + 1] = P_(1,1); msg.pose.covariance[1*6 + 5] = P_(1,6);
    msg.pose.covariance[5*6 + 0] = P_(6,0); msg.pose.covariance[5*6 + 1] = P_(6,1); msg.pose.covariance[5*6 + 5] = P_(6,6);

    msg.twist.twist.linear.x = x_(2);
    msg.twist.twist.linear.y = x_(3);
    msg.twist.twist.angular.z = x_(7);

    // Maps [vx, vy, omega] from P_ to 6x6 twist covariance
    msg.twist.covariance.fill(0.0);
    msg.twist.covariance[0*6 + 0] = P_(2,2); msg.twist.covariance[0*6 + 1] = P_(2,3); msg.twist.covariance[0*6 + 5] = P_(2,7);
    msg.twist.covariance[1*6 + 0] = P_(3,2); msg.twist.covariance[1*6 + 1] = P_(3,3); msg.twist.covariance[1*6 + 5] = P_(3,7);
    msg.twist.covariance[5*6 + 0] = P_(7,2); msg.twist.covariance[5*6 + 1] = P_(7,3); msg.twist.covariance[5*6 + 5] = P_(7,7);

    fused_odom_pub_->publish(msg);

    // Broadcast the transform from odom -> base_link
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped.header.stamp = msg.header.stamp;
    transform_stamped.header.frame_id = odom_frame_id_;
    transform_stamped.child_frame_id = base_frame_id_;
    transform_stamped.transform.translation.x = x_(0);
    transform_stamped.transform.translation.y = x_(1);
    transform_stamped.transform.translation.z = 0.0;
    transform_stamped.transform.rotation = tf2::toMsg(q);
    
    tf_broadcaster_->sendTransform(transform_stamped);
}

void SaganKalmanFilter::reset_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    (void)request;
    RCLCPP_INFO(this->get_logger(), "Resetting Kalman filter state.");
    x_ = StateVector::Zero();
    P_ = StateCovariance::Identity() * 1000.0;
    last_predict_time_ = this->get_clock()->now();
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

