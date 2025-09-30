#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from nav_msgs.msg import Odometry
import math
import numpy as np
import time

class SaganKalmanFilterOptimizer(Node):
    def __init__(self):
        super().__init__('sagan_kalman_filter_optimizer')

        # --- GA Parameters (Tune these) ---
        self.population_size = 20
        self.num_generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.gene_min_val = 0.01      # Min value for any covariance element
        self.gene_max_val = 1000.0  # Max value for any covariance element

        # --- Dynamic Configuration Variables ---
        self.num_genes = None # Will be determined dynamically
        self.q_size = 0
        self.odom_r_size = 0
        self.imu_r_size = 0

        # --- ROS 2 Clients & Publishers ---
        self.param_client = AsyncParameterClient(self, "/sagan_kalman_filter")
        self.reset_gz_pose_client = self.create_client(SetEntityPose, '/world/default/set_pose')
        self.reset_odom_client = self.create_client(Trigger, 'reset_odometry')
        self.reset_ekf_client = self.create_client(Trigger, 'reset_ekf')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- ROS 2 Subscribers for Fitness Evaluation ---
        self.ground_truth_topic = "/odom_gz"
        self.ekf_topic = "/odom/filtered"
        self.noisy_odom_topic = "/odom/with_noise"

        self.ground_truth_poses = []
        self.ekf_poses = []
        self.noisy_poses = [] 

        self.ground_truth_sub = self.create_subscription(Odometry, self.ground_truth_topic, self.ground_truth_callback, 10)
        self.ekf_sub = self.create_subscription(Odometry, self.ekf_topic, self.ekf_callback, 10)
        self.noisy_sub = self.create_subscription(Odometry, self.noisy_odom_topic, self.noisy_callback, 10)

        self.get_logger().info("Kalman Filter Optimizer is ready.")

    # --- Subscriber Callbacks ---
    def ground_truth_callback(self, msg):
        self.ground_truth_poses.append(msg.pose.pose)

    def ekf_callback(self, msg):
        self.ekf_poses.append(msg.pose.pose)

    def noisy_callback(self, msg):
        self.noisy_poses.append(msg.pose.pose)

    def setup_optimizer_sync(self):
        """
        Fetches parameters from the EKF node to configure the optimizer dynamically.
        This makes the optimizer adaptable to changes in the EKF's sensor configuration.
        """
        self.get_logger().info("Waiting for EKF parameter service...")
        if not self.param_client.wait_for_services(timeout_sec=5.0):
            self.get_logger().error("Parameter service '/sagan_kalman_filter/get_parameters' not available. Exiting.")
            return False

        self.get_logger().info("Fetching EKF configuration to adapt optimizer...")
        params_to_get = ['Q_diag', 'odom_sensor.measurement_map', 'imu_sensor.measurement_map']
        future = self.param_client.get_parameters(params_to_get)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response is None:
            self.get_logger().error("Failed to get EKF parameters. Is the filter node running?")
            return False
            
        # The response contains a list of ParameterValue objects in the same order as the request.
        # We zip the requested names with the returned values to create a dictionary.
        param_dict = {}
        for name, value_msg in zip(params_to_get, response.values):
            if value_msg.type == Parameter.Type.NOT_SET.value:
                self.get_logger().error(f"Parameter '{name}' was not set on the EKF node.")
                return False
            # Extract the Python value from the ParameterValue message
            if value_msg.type == Parameter.Type.DOUBLE_ARRAY.value:
                 param_dict[name] = list(value_msg.double_array_value)
            elif value_msg.type == Parameter.Type.BOOL_ARRAY.value:
                 param_dict[name] = list(value_msg.bool_array_value)
            else:
                 # Add other types if you need them in the future
                 self.get_logger().warn(f"Unhandled parameter type for {name}")

        # Validate that all required parameters were fetched and added to the dictionary
        for param_name in params_to_get:
            if param_name not in param_dict:
                self.get_logger().error(f"Parameter '{param_name}' not found on the EKF node. Is it running with the correct yaml file?")
                return False

        # Calculate the number of genes for each covariance matrix
        self.q_size = len(param_dict['Q_diag'])
        self.odom_r_size = sum(param_dict['odom_sensor.measurement_map'])
        self.imu_r_size = sum(param_dict['imu_sensor.measurement_map'])
        
        self.num_genes = self.q_size + self.odom_r_size + self.imu_r_size
        
        self.get_logger().info("Optimizer configured dynamically:")
        self.get_logger().info(f"  - Q_diag size: {self.q_size}")
        self.get_logger().info(f"  - odom_sensor.R_diag size: {self.odom_r_size}")
        self.get_logger().info(f"  - imu_sensor.R_diag size: {self.imu_r_size}")
        self.get_logger().info(f"  -> Total genes to optimize: {self.num_genes}")

        if self.num_genes == 0:
            self.get_logger().error("Total number of genes is zero. Check your EKF configuration.")
            return False
        
        return True

    # --- Core GA Functions ---
    def run_optimizer(self):
        """Main genetic algorithm loop."""
        # Configure the optimizer based on the running EKF node's parameters
        if not self.setup_optimizer_sync():
            self.get_logger().error("Optimizer setup failed. Shutting down.")
            return
            
        self.get_logger().info("Starting Genetic Algorithm...")
        population = self.initialize_population()

        for generation in range(self.num_generations):
            self.get_logger().info(f"--- Generation {generation + 1}/{self.num_generations} ---")

            fitness_scores = [self.evaluate_fitness(ind) for ind in population]

            best_fitness = np.max(fitness_scores)
            best_individual_idx = np.argmax(fitness_scores)
            best_individual = population[best_individual_idx]
            
            self.get_logger().info(f"Generation Best Fitness: {best_fitness:.4f}")
            self.log_individual(best_individual)

            parents = self.selection(population, fitness_scores)
            
            next_population = []
            while len(next_population) < self.population_size:
                parent1, parent2 = parents[np.random.randint(0, len(parents))], parents[np.random.randint(0, len(parents))]
                if np.random.rand() < self.crossover_rate:
                    offspring = self.crossover(parent1, parent2)
                else:
                    offspring = parent1.copy()
                
                next_population.append(self.mutate(offspring))

            population = np.array(next_population)

        self.get_logger().info("--- Genetic Algorithm Finished ---")
        final_fitness_scores = [self.evaluate_fitness(ind) for ind in population]
        final_best_idx = np.argmax(final_fitness_scores)
        final_best_chromosome = population[final_best_idx]
        self.get_logger().info("Optimal Covariances Found:")
        self.log_individual(final_best_chromosome)

    def evaluate_fitness(self, individual):
        """
        Runs a single trial and returns a fitness score based on both
        accuracy and improvement over noisy odometry.
        """
        self.set_kalman_params(individual)
        time.sleep(0.5)

        self.reset_simulation_state()
        self.ground_truth_poses.clear()
        self.ekf_poses.clear()
        self.noisy_poses.clear()
        time.sleep(0.5)

        self.execute_defined_path()

        if not self.ground_truth_poses or not self.ekf_poses or not self.noisy_poses:
            self.get_logger().error("Did not receive pose data on all required topics, returning fitness of 0.")
            return 0.0

        min_len = min(len(self.ground_truth_poses), len(self.ekf_poses), len(self.noisy_poses))
        
        gt = np.array([[p.position.x, p.position.y] for p in self.ground_truth_poses[:min_len]])
        ekf = np.array([[p.position.x, p.position.y] for p in self.ekf_poses[:min_len]])
        noisy = np.array([[p.position.x, p.position.y] for p in self.noisy_poses[:min_len]])

        rmse_filtered_vs_truth = np.sqrt(np.mean((gt - ekf)**2))
        rmse_noisy_vs_truth = np.sqrt(np.mean((gt - noisy)**2))
        
        if rmse_filtered_vs_truth < 1e-6:
            rmse_filtered_vs_truth = 1e-6

        accuracy_score = 1.0 / (1.0 + rmse_filtered_vs_truth)
        improvement_ratio = rmse_noisy_vs_truth / rmse_filtered_vs_truth
        fitness = improvement_ratio
        self.get_logger().info(f"RMSE filtered_vs_truth: {rmse_filtered_vs_truth:.4f}")
        self.get_logger().info(f"RMSE noisy_vs_truth: {rmse_noisy_vs_truth:.4f}")
        self.get_logger().info(f"Accuracy score: {accuracy_score:.4f}")
        self.get_logger().info(f"Improvement ratio: {improvement_ratio:.4f}")
        self.get_logger().info(f"Fitness: {fitness:.4f}")


        return fitness

    # --- GA Operators ---
    def initialize_population(self):
        return np.random.uniform(low=self.gene_min_val, high=self.gene_max_val, size=(self.population_size, self.num_genes))

    def selection(self, population, fitness_scores):
        """Tournament selection."""
        selected = []
        for _ in range(self.population_size):
            i1, i2 = np.random.choice(len(population), 2, replace=False)
            winner = i1 if fitness_scores[i1] > fitness_scores[i2] else i2
            selected.append(population[winner])
        return selected

    def crossover(self, parent1, parent2):
        """Single-point crossover."""
        point = np.random.randint(1, self.num_genes - 1)
        return np.concatenate([parent1[:point], parent2[point:]])

    def mutate(self, individual):
        """Randomly tweak genes."""
        for i in range(self.num_genes):
            if np.random.rand() < self.mutation_rate:
                change = np.random.uniform(-0.1, 0.1) * individual[i]
                individual[i] = np.clip(individual[i] + change, self.gene_min_val, self.gene_max_val)
        return individual

    # --- Helper Functions ---
    def set_kalman_params(self, individual):
        """Slices the chromosome dynamically and sets the covariance parameters."""
        # Calculate slice indices
        end_q = self.q_size
        end_odom_r = end_q + self.odom_r_size
        
        # Slice the individual's chromosome
        q_diag = individual[0:end_q].tolist()
        odom_sensor_r_diag = individual[end_q:end_odom_r].tolist()
        imu_sensor_r_diag = individual[end_odom_r:].tolist()

        params_to_set = []
        if self.q_size > 0:
            params_to_set.append(Parameter('Q_diag', Parameter.Type.DOUBLE_ARRAY, q_diag))
        if self.odom_r_size > 0:
            params_to_set.append(Parameter('odom_sensor.R_diag', Parameter.Type.DOUBLE_ARRAY, odom_sensor_r_diag))
        if self.imu_r_size > 0:
            params_to_set.append(Parameter('imu_sensor.R_diag', Parameter.Type.DOUBLE_ARRAY, imu_sensor_r_diag))

        if not params_to_set:
            self.get_logger().warn("No parameters to set. Check gene sizes.")
            return

        future = self.param_client.set_parameters(params_to_set)
        rclpy.spin_until_future_complete(self, future)

    def reset_simulation_state(self):
        """Resets Gazebo pose and odometry/EKF nodes."""
        gz_req = SetEntityPose.Request()
        gz_req.entity.name = "Sagan"
        gz_req.entity.type = Entity.MODEL
        gz_req.pose.position = Point(x=0.0, y=0.0, z=0.2)
        gz_req.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.reset_gz_pose_client.call_async(gz_req)

        reset_req = Trigger.Request()
        self.reset_odom_client.call_async(reset_req)
        self.reset_ekf_client.call_async(reset_req)

    def execute_defined_path(self):
        """Commands the robot to drive along a path."""
        self.publish_velocity(linear=0.6, angular=0.3)
        self.spin_for_duration(5.0)
        self.publish_velocity(linear=0.6, angular=-0.4)
        self.spin_for_duration(5.0)
        self.publish_velocity(linear=0.6, angular=0.0)
        self.spin_for_duration(5.0)
        self.publish_velocity(linear=0.8, angular=0.1)
        self.spin_for_duration(5.0)
        self.publish_velocity(linear=0.0, angular=0.0)
        time.sleep(1)

    def publish_velocity(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)

    def spin_for_duration(self, duration_sec):
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration_sec:
            rclpy.spin_once(self, timeout_sec=0.01)

    def log_individual(self, individual):
        """Logs the individual's genes using dynamic slicing."""
        # Calculate slice indices
        end_q = self.q_size
        end_odom_r = end_q + self.odom_r_size

        # Slice and round for logging
        q = np.round(individual[0:end_q], 4).tolist()
        r_o = np.round(individual[end_q:end_odom_r], 4).tolist()
        r_i = np.round(individual[end_odom_r:], 4).tolist()

        self.get_logger().info(f"    Q_diag ({self.q_size} genes): {q}")
        self.get_logger().info(f"    odom_sensor.R_diag ({self.odom_r_size} genes): {r_o}")
        self.get_logger().info(f"    imu_sensor.R_diag ({self.imu_r_size} genes): {r_i}")


def main(args=None):
    rclpy.init(args=args)
    optimizer_node = SaganKalmanFilterOptimizer()

    optimizer_node.run_optimizer()
    
    optimizer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

