#include <chrono>
#include <memory>
#include <algorithm>
#include "rclcpp/rclcpp.hpp"
#include "sagan_interfaces/msg/sagan_cmd.hpp"
#include "geometry_msgs/msg/twist.hpp"

using namespace std::chrono_literals;

class SaganDifferentialDriver : public rclcpp::Node
{
public:
    SaganDifferentialDriver()
        : Node("sagan_differential_drive")
    {
        RCLCPP_INFO(this->get_logger(), "Starting Twist to Wheels translator...");

        // --- Parameters ---
        this->declare_parameter<double>("wheel_radius", 0.060);
        this->declare_parameter<double>("wheel_separation", 0.370);

        // Read parameters
        this->get_parameter("wheel_radius", wheel_radius_);
        this->get_parameter("wheel_separation", wheel_separation_);

        RCLCPP_INFO(this->get_logger(), "Using wheel radius: %.4f m", wheel_radius_);
        RCLCPP_INFO(this->get_logger(), "Using wheel separation: %.4f m", wheel_separation_);

        publisher_ = this->create_publisher<sagan_interfaces::msg::SaganCmd>("/SaganCommands", 1);

        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&SaganDifferentialDriver::cmd_vel_callback, this, std::placeholders::_1));

    }

private:
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        // Extract linear and angular velocities
        double linear_vel = msg->linear.x;
        double angular_vel = msg->angular.z;

        // --- Inverse Kinematics Calculation ---
        // See Python file for detailed formula explanation.
        double left_wheel_vel = (linear_vel + (angular_vel * wheel_separation_ / 2.0)) / wheel_radius_;
        double right_wheel_vel = (linear_vel - (angular_vel * wheel_separation_ / 2.0)) / wheel_radius_;

        auto message = sagan_interfaces::msg::SaganCmd();
        
        for (int index = 0; index < 3; index = index + 2)
        {
            message.wheel_cmd[index + 1].angular_velocity = right_wheel_vel;
            message.wheel_cmd[index].angular_velocity = left_wheel_vel;
        }        
        // Publish the message
        publisher_->publish(message);

    }
    rclcpp::Publisher<sagan_interfaces::msg::SaganCmd>::SharedPtr publisher_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

    double wheel_radius_;
    double wheel_separation_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SaganDifferentialDriver>());
    rclcpp::shutdown();
    return 0;
}
