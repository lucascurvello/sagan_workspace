import rclpy
from rclpy.node import Node
from rclpy.client import Client
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose, Point, Quaternion
import math


class SaganKalmanFilterOptimizer(Node):
    def __init__(self):
        super().__init__('sagan_kalman_filter_optimizer')
        
        # The name of the node we want to change
        target_node_name = "/sagan_kalman_filter" # Use the fully qualified node name

        # Create a synchronous client
        # The client's node does not need to spin_once() for the Sync Client
        self.param_client = AsyncParameterClient(self, target_node_name)

        # Wait for the service to be available
        if not self.param_client.wait_for_services(timeout_sec=1.0):
            self.get_logger().error(f"Service for node '{target_node_name}' not available. Exiting.")
            return

    def reset_model_position(self, model_name, x, y, z, yaw_degrees):

        # Reset gazebo position service client
        self.reset_position = self.create_client(SetEntityPose, '/world/default/set_pose')
        
        while not self.reset_position.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        request = SetEntityPose.Request()
        
        # Reset odometry service client
        self.client_reset_odometry = self.create_client(Trigger, 'reset_odometry')

        while not self.client_reset_odometry.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        self.request_reset_odometry = Trigger.Request()

        # Reset ekf service client
        self.client_reset_ekf = self.create_client(Trigger, 'reset_ekf')

        while not self.client_reset_ekf.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        self.request_reset_ekf = Trigger.Request()

        """
        Calls the Gazebo service to set the model's state.
        :param model_name: The name of the model to reset.
        :param x: The new x-coordinate.
        :param y: The new y-coordinate.
        :param z: The new z-coordinate.
        :param yaw_degrees: The new yaw angle in degrees.
        """
        self.get_logger().info(f"Attempting to reset position for model: '{model_name}'")

        # 1. Create an instance of the Entity message
        entity_msg = Entity()
        # 2. Set the 'name' field of the message object
        entity_msg.name = model_name
        entity_msg.type = Entity.MODEL
        # 3. Assign the entire object to the request field
        request.entity = entity_msg
  
        # Set the pose
        request.pose.position = Point(x=x, y=y, z=z)
        
        # Convert yaw angle (degrees) to a quaternion
        yaw_radians = math.radians(yaw_degrees)
        q = self.euler_to_quaternion(0, 0, yaw_radians)
        request.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        self.client_reset_ekf.call_async(self.request_reset_ekf)

        self.client_reset_odometry.call_async(self.request_reset_odometry)

        return self.reset_position.call_async(request)
    
    def set_params(self):
        # Define the new parameter value
        new_param = Parameter('Q_diag', Parameter.Type.DOUBLE_ARRAY, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        # Set the parameter
        self.param_client.set_parameters([new_param])



    def euler_to_quaternion(self, roll, pitch, yaw):
        """Converts Euler angles to quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [x, y, z, w]

    def response_callback(self, future):
        """Callback to handle the service response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Model position reset successfully.')
            else:
                self.get_logger().error(f'Failed to reset model position: {response.status_message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = SaganKalmanFilterOptimizer()
    node.reset_model_position('Sagan', 0.0, 0.0, 0.2, 0.0)
    node.set_params()
    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
