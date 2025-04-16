#include <rclcpp/rclcpp.hpp>
#include <sagan_interfaces/msg/sagan_states.hpp> // Custom message for SaganStates
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <chrono>
#include <string>

using namespace std::chrono_literals;

class SaganOdometryNode : public rclcpp::Node
{
public:
    SaganOdometryNode()
    : Node("sagan_odometry_node"),
      x_(0.0), y_(0.0), theta_(0.0), last_x_(0.0), last_y_(0.0), last_theta_(0.0)  // Initialize robot position and orientation
    {
        // Subscriber to SaganStates topic
        subscription_ = this->create_subscription<sagan_interfaces::msg::SaganStates>(
            "/Sagan/SaganStates", 10, std::bind(&SaganOdometryNode::state_callback, this, std::placeholders::_1));

        // Publisher to odom topic
        odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);

        // Initialize transform broadcaster
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // Timer for regular odom publishing
        timer_ = this->create_wall_timer(10ms, std::bind(&SaganOdometryNode::timer_callback, this));
    }

private:
    void state_callback(const sagan_interfaces::msg::SaganStates::SharedPtr msg)
    {
        for (auto index = 0; index < 4; index++)
        {
            omega[index] = msg->wheel_state[index].angular_velocity;
            delta[index] = msg->steering_state[index].angular_position;
        }
    }

    void timer_callback()
    {
        double wheel_radius = 0.06;
        double wheel_base = 0.370; 
        double wheel_separetion = 0.2975;

        double delta_t = 0.01;  // Time step (could be obtained dynamically or use a constant)

        double x_sep[4] = {-wheel_base/2, -wheel_base/2, wheel_base/2, wheel_base/2};
        double y_sep[4] = {-wheel_separetion/2, wheel_separetion/2, -wheel_separetion/2, wheel_separetion/2};

        //Update last values
        last_x_ = x_;
        last_y_ = y_;
        last_theta_ = theta_;
        
        double v_theta = 0;
        for (auto i = 0; i < 4; i++)
        {
            v_theta += omega[i] * wheel_radius * (-y_sep[i] * cos(delta[i]) + x_sep[i] * sin(delta[i])) / (4 * (x_sep[i] * x_sep[i]) + 4 * (y_sep[i] * y_sep[i]));
        }

        theta_ += 2.04203 * 1.002758041 * v_theta * delta_t;

        double vx = 0;
        double vy = 0;
        for (auto index = 0; index < 4; index++)
        {
            vx += omega[index] * wheel_radius * cos(delta[index] + theta_) / 4;
            vy += omega[index] * wheel_radius * sin(delta[index] + theta_) / 4;
        }

        // Update robot pose
        x_ += vx * delta_t;
        y_ += vy * delta_t;

        // Publish Odometry Message
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = this->now();
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "base_footprint";

        // Set position
        odom_msg.pose.pose.position.x = x_;
        odom_msg.pose.pose.position.y = y_;
        odom_msg.pose.pose.position.z = 0.0;

        odom_msg.twist.twist.linear.x = (x_ - last_x_) / 0.01;
        odom_msg.twist.twist.linear.y = (y_ - last_y_) / 0.01;
        odom_msg.twist.twist.angular.z = (theta_ - last_theta_) / 0.01;

        // Set orientation (convert theta to quaternion) 
        tf2::Quaternion q;
        q.setRPY(0, 0, theta_);
        q.normalize();
        geometry_msgs::msg::Quaternion msg_quat = tf2::toMsg(q);
        odom_msg.pose.pose.orientation = msg_quat;

        // Set velocities
        //odom_msg.twist.twist.linear.x = SaganOdometryNode::last_v_;
        //odom_msg.twist.twist.angular.z = SaganOdometryNode::last_theta_;

        // Broadcast Transform from odom to base_link
        geometry_msgs::msg::TransformStamped transform_stamped;
        transform_stamped.header.stamp = this->get_clock()->now();
        transform_stamped.header.frame_id = "odom";
        transform_stamped.child_frame_id = "base_footprint";
        transform_stamped.transform.translation.x = x_;
        transform_stamped.transform.translation.y = y_;
        transform_stamped.transform.translation.z = 0.0;
        transform_stamped.transform.rotation= msg_quat;

        // Send the transform
        tf_broadcaster_->sendTransform(transform_stamped);
        odom_publisher_->publish(odom_msg);
    }

    // Member variables
    rclcpp::Subscription<sagan_interfaces::msg::SaganStates>::SharedPtr subscription_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    double x_, y_, theta_;   // Robot pose
    double last_x_, last_y_, last_theta_;  // Last computed velocities
    double omega[4], delta[4];
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SaganOdometryNode>());
    rclcpp::shutdown();
    return 0;
}
