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

        timer_ = this->create_wall_timer(10ms, std::bind(&MinimalPublisher::publish_message, this));
    }

private:
    void publish_message()
    {
        auto message = sagan_interfaces::msg::SaganCmd();

        for (int index = 0; index < 4; index++)
        {
            
            message.steering_cmd[index].angular_position = 0.0*3.14/180;
        }  
        
        for (int index = 0; index < 3; index = index + 2)
        {
            message.wheel_cmd[index + 1].angular_velocity = 8.0;
            message.wheel_cmd[index].angular_velocity = 10.0;
        }        
        // if (MinimalPublisher::x == 0){
        //     for (int index = 0; index < 4; index++)
        //     {
        //         message.wheel_cmd[index].angular_velocity = 10.0;
        //         message.steering_cmd[index].angular_position = 30*3.14/180;
        //     }   
        //     MinimalPublisher::x = 1;
        // }else{
        //     for (int index = 0; index < 4; index++)
        //     {
        //         message.wheel_cmd[index].angular_velocity = 10.0;
        //         message.steering_cmd[index].angular_position = 30*3.14/180;
        //     }  
        // }

        // message.steering_cmd[0].angular_position = 0*3.14/180;
        // message.steering_cmd[1].angular_position = 30*3.14/180;
        // message.steering_cmd[2].angular_position = 60*3.14/180;
        // message.steering_cmd[3].angular_position = 90*3.14/180;

        // Publish the message
        publisher_->publish(message);

    }

    int x = 0;
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
