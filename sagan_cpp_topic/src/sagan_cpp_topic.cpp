#include <chrono>
#include <memory>
#include <algorithm>
#include "rclcpp/rclcpp.hpp"
#include "sagan_interfaces/msg/sagan_cmd.hpp"

using namespace std::chrono_literals;

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher()
        : Node("sagan_cmd")
    {
        publisher_ = this->create_publisher<sagan_interfaces::msg::SaganCmd>("/SaganCommands", 1);

        timer_ = this->create_wall_timer(100ms, std::bind(&MinimalPublisher::publish_message, this));
    }

private:
    void publish_message()
    {
        // Create a message object
        auto message = sagan_interfaces::msg::SaganCmd();
        
        for (int index = 0; index < 4; index++)
        {
            message.wheel_cmd[index].angular_velocity = 10.0;
            message.steering_cmd[index].angular_position = 10.0;
        }
        
        // Publish the message
        publisher_->publish(message);

    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sagan_interfaces::msg::SaganCmd>::SharedPtr publisher_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalPublisher>());
    rclcpp::shutdown();
    return 0;
}
