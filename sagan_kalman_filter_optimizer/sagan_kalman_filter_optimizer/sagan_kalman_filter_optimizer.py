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
        self.num_generations = 10
        self.mutation_rate = 0.3
        self.crossover_rate = 0.9
        # Total genes = 8 (Q_diag) + 6 (R_odom_diag) + 3 (R_imu_diag)
        self.num_genes = 17
        self.gene_min_val = 0.01 # Min value for any covariance element
        self.gene_max_val = 100000.0  # Max value for any covariance element

        # --- ROS 2 Clients & Publishers ---
        self.param_client = AsyncParameterClient(self, "/sagan_kalman_filter")
        self.reset_gz_pose_client = self.create_client(SetEntityPose, '/world/default/set_pose')
        self.reset_odom_client = self.create_client(Trigger, 'reset_odometry')
        self.reset_ekf_client = self.create_client(Trigger, 'reset_ekf')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- ROS 2 Subscribers for Fitness Evaluation ---
        # ❗️ IMPORTANT: Change these topic names to match your robot's setup!
        self.ground_truth_topic = "/odom_gz"  # From a Gazebo plugin, for example
        self.ekf_topic = "/odom/filtered"          # Your Kalman filter's output
        self.noisy_odom_topic = "/odom/with_noise" # Your new noisy topic

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

    # --- Core GA Functions ---
    def run_optimizer(self):
        """Main genetic algorithm loop."""
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
        final_best_idx = np.argmax(fitness_scores)
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
        # Clear all three lists before the trial
        self.ground_truth_poses.clear()
        self.ekf_poses.clear()
        self.noisy_poses.clear()
        time.sleep(0.5)

        self.execute_defined_path()

        # Check if we received data on all three topics
        if not self.ground_truth_poses or not self.ekf_poses or not self.noisy_poses:
            self.get_logger().error("Did not receive pose data on all required topics, returning fitness of 0.")
            return 0.0

        # Calculate error for both filtered and noisy data against the ground truth
        min_len = min(len(self.ground_truth_poses), len(self.ekf_poses), len(self.noisy_poses))
        
        gt = np.array([[p.position.x, p.position.y] for p in self.ground_truth_poses[:min_len]])
        ekf = np.array([[p.position.x, p.position.y] for p in self.ekf_poses[:min_len]])
        noisy = np.array([[p.position.x, p.position.y] for p in self.noisy_poses[:min_len]])

        # Calculate RMSE for both
        rmse_filtered_vs_truth = np.sqrt(np.mean((gt - ekf)**2))
        rmse_noisy_vs_truth = np.sqrt(np.mean((gt - noisy)**2))
        
        # Avoid division by zero if an error is 0
        if rmse_filtered_vs_truth < 1e-6:
             rmse_filtered_vs_truth = 1e-6

        # --- New Fitness Calculation ---
        accuracy_score = 1.0 / (1.0 + rmse_filtered_vs_truth)
        improvement_ratio = rmse_noisy_vs_truth / rmse_filtered_vs_truth

        # The final fitness is a product of accuracy and improvement
        fitness = accuracy_score * improvement_ratio

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
        """Slices the chromosome and sets the three covariance parameters."""
        q_diag = individual[0:8].tolist()
        r_odom_diag = individual[8:14].tolist()
        r_imu_diag = individual[14:17].tolist()

        params_to_set = [
            Parameter('Q_diag', Parameter.Type.DOUBLE_ARRAY, q_diag),
            Parameter('R_odom_diag', Parameter.Type.DOUBLE_ARRAY, r_odom_diag),
            Parameter('R_imu_diag', Parameter.Type.DOUBLE_ARRAY, r_imu_diag)
        ]
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
        """Commands the robot to drive in a 1m x 1m square."""
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
        q = np.round(individual[0:8], 4).tolist()
        r_o = np.round(individual[8:14], 4).tolist()
        r_i = np.round(individual[14:17], 4).tolist()
        self.get_logger().info(f"    Q_diag: {q}")
        self.get_logger().info(f"    R_odom_diag: {r_o}")
        self.get_logger().info(f"    R_imu_diag: {r_i}")


def main(args=None):
    rclpy.init(args=args)
    optimizer_node = SaganKalmanFilterOptimizer()

    # Wait for services before starting
    if not optimizer_node.param_client.wait_for_services(timeout_sec=5.0):
        optimizer_node.get_logger().error("Parameter service not available. Exiting.")
        return

    optimizer_node.run_optimizer()
    optimizer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()