#include "sagan_odometry/sagan_odometry.hpp"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>

using namespace std::chrono_literals;

SaganOdometryNode::SaganOdometryNode()
: Node("sagan_odometry_node"),
  x_(0.0), y_(0.0), theta_(0.0),
  noisy_x_(0.0), noisy_y_(0.0), noisy_theta_(0.0),
  random_generator_(std::random_device{}()), 
  noise_distribution_(0.0, 1.0)
{
    // Initialize member arrays to zero
    for(int i=0; i<4; ++i) {
        omega_clean_[i] = 0.0;
        omega_noisy_[i] = 0.0;
        delta_[i] = 0.0;
    }

    this->declare_parameter<double>("wheel_velocity_noise_scaler", 0.02);
    this->get_parameter("wheel_velocity_noise_scaler", wheel_velocity_noise_scaler_);

    subscription_ = this->create_subscription<sagan_interfaces::msg::SaganStates>(
        "/Sagan/SaganStates", 10, std::bind(&SaganOdometryNode::state_callback, this, std::placeholders::_1));

    odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
    
    odom_with_noise_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom/with_noise", 10);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    timer_ = this->create_wall_timer(10ms, std::bind(&SaganOdometryNode::timer_callback, this));

    last_time_ = this->now();
}

void SaganOdometryNode::state_callback(const sagan_interfaces::msg::SaganStates::SharedPtr msg)
{
    for (auto index = 0; index < 4; index++)
    {
        omega_clean_[index] = msg->wheel_state[index].angular_velocity;
        delta_[index] = msg->steering_state[index].angular_position;

        double std_dev = std::sqrt(wheel_velocity_noise_scaler_ * std::abs(omega_clean_[index]));
        double noise = noise_distribution_(random_generator_) * std_dev;

        omega_noisy_[index] = omega_clean_[index] + noise;
    }
}

void SaganOdometryNode::timer_callback()
{
    rclcpp::Time current_time = this->now();
    double delta_t = (current_time - last_time_).seconds();
    if (delta_t <= 1e-9) {
        return;
    }
    last_time_ = current_time;

    double wheel_radius = 0.06;
    double wheel_base = 0.370; 
    double slipage_coefficient = 0.166508;

    // --- 1. CLEAN ODOMETRY ---
    {
        double v_theta = wheel_radius * (omega_clean_[0] - omega_clean_[1]) / wheel_base;
        double v_forward = wheel_radius * (omega_clean_[0] + omega_clean_[1]) / 2.0;
        double v_lateral = slipage_coefficient * v_theta;

        theta_ += v_theta * delta_t;
        double vx_world = v_forward * cos(theta_) + v_lateral * sin(theta_);
        double vy_world = v_forward * sin(theta_) - v_lateral * cos(theta_);
        x_ += vx_world * delta_t;
        y_ += vy_world * delta_t;
        
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = this->now();
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "base_footprint";
        
        odom_msg.pose.pose.position.x = x_;
        odom_msg.pose.pose.position.y = y_;
        tf2::Quaternion q;
        q.setRPY(0, 0, theta_);
        odom_msg.pose.pose.orientation = tf2::toMsg(q);

        odom_msg.twist.twist.linear.x = v_forward;
        odom_msg.twist.twist.linear.y = -v_lateral;
        odom_msg.twist.twist.angular.z = v_theta;
        
        odom_publisher_->publish(odom_msg);

        geometry_msgs::msg::TransformStamped transform_stamped;
        transform_stamped.header.stamp = this->get_clock()->now();
        transform_stamped.header.frame_id = "odom";
        transform_stamped.child_frame_id = "base_footprint";
        transform_stamped.transform.translation.x = x_;
        transform_stamped.transform.translation.y = y_;
        transform_stamped.transform.rotation = tf2::toMsg(q);
        tf_broadcaster_->sendTransform(transform_stamped);
    }

    // --- 2. NOISY ODOMETRY ---
    {
        double noisy_v_theta = wheel_radius * (omega_noisy_[0] - omega_noisy_[1]) / wheel_base;
        double noisy_v_forward = wheel_radius * (omega_noisy_[0] + omega_noisy_[1]) / 2.0;
        double noisy_v_lateral = slipage_coefficient * noisy_v_theta;

        noisy_theta_ += noisy_v_theta * delta_t;
        double noisy_vx_world = noisy_v_forward * cos(noisy_theta_) + noisy_v_lateral * sin(noisy_theta_);
        double noisy_vy_world = noisy_v_forward * sin(noisy_theta_) - noisy_v_lateral * cos(noisy_theta_);
        noisy_x_ += noisy_vx_world * delta_t;
        noisy_y_ += noisy_vy_world * delta_t;

        auto noisy_odom_msg = nav_msgs::msg::Odometry();
        noisy_odom_msg.header.stamp = this->now();
        noisy_odom_msg.header.frame_id = "odom";
        noisy_odom_msg.child_frame_id = "noise_odometry";

        noisy_odom_msg.pose.pose.position.x = noisy_x_;
        noisy_odom_msg.pose.pose.position.y = noisy_y_;
        tf2::Quaternion q_noisy;
        q_noisy.setRPY(0, 0, noisy_theta_);
        noisy_odom_msg.pose.pose.orientation = tf2::toMsg(q_noisy);

        noisy_odom_msg.twist.twist.linear.x = noisy_v_forward;
        noisy_odom_msg.twist.twist.linear.y = -noisy_v_lateral;
        noisy_odom_msg.twist.twist.angular.z = noisy_v_theta;

        odom_with_noise_publisher_->publish(noisy_odom_msg);

        geometry_msgs::msg::TransformStamped transform_stamped;
        transform_stamped.header.stamp = this->get_clock()->now();
        transform_stamped.header.frame_id = "odom";
        transform_stamped.child_frame_id = "noise_odometry";
        transform_stamped.transform.translation.x = noisy_x_;
        transform_stamped.transform.translation.y = noisy_y_;
        transform_stamped.transform.rotation = tf2::toMsg(q_noisy);
        tf_broadcaster_->sendTransform(transform_stamped);
    }
}

// --- MAIN FUNCTION ---
// The entry point for the ROS 2 node.
int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    // Create an instance of the SaganOdometryNode and spin it
    rclcpp::spin(std::make_shared<SaganOdometryNode>());
    rclcpp::shutdown();
    return 0;
}

