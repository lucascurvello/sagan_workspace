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
        self.elitism_count = 1 # Number of best individuals to carry over
        self.gene_min_val = 0.001     # Min value for any covariance element
        self.gene_max_val = 1000.0     # Max value for any covariance element

        # --- Dynamic Configuration Variables ---
        self.num_genes = None
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

        # Data lists now store (timestamp, pose) tuples
        self.ground_truth_data = []
        self.ekf_data = []
        self.noisy_data = []

        self.ground_truth_sub = self.create_subscription(Odometry, self.ground_truth_topic, self.ground_truth_callback, 10)
        self.ekf_sub = self.create_subscription(Odometry, self.ekf_topic, self.ekf_callback, 10)
        self.noisy_sub = self.create_subscription(Odometry, self.noisy_odom_topic, self.noisy_callback, 10)

        self.get_logger().info("Kalman Filter Optimizer is ready.")

    # --- Subscriber Callbacks ---
    def ground_truth_callback(self, msg):
        time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.ground_truth_data.append((time_sec, msg.pose.pose))

    def ekf_callback(self, msg):
        time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.ekf_data.append((time_sec, msg.pose.pose))

    def noisy_callback(self, msg):
        time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.noisy_data.append((time_sec, msg.pose.pose))

    def setup_optimizer_sync(self):
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

        param_dict = {}
        for name, value_msg in zip(params_to_get, response.values):
            if value_msg.type == Parameter.Type.DOUBLE_ARRAY.value:
                param_dict[name] = list(value_msg.double_array_value)
            elif value_msg.type == Parameter.Type.BOOL_ARRAY.value:
                param_dict[name] = list(value_msg.bool_array_value)
        
        for p in params_to_get:
            if p not in param_dict:
                self.get_logger().error(f"Parameter '{p}' not found. Check EKF yaml.")
                return False

        self.q_size = len(param_dict['Q_diag'])
        self.odom_r_size = sum(param_dict['odom_sensor.measurement_map'])
        self.imu_r_size = sum(param_dict['imu_sensor.measurement_map'])
        self.num_genes = self.q_size + self.odom_r_size + self.imu_r_size

        self.get_logger().info("Optimizer configured dynamically:")
        self.get_logger().info(f"  -> Total genes to optimize: {self.num_genes}")
        return True

    def align_and_get_poses(self):
        """
        Aligns EKF and noisy poses to the ground truth timeline based on the closest timestamp
        and returns three numpy arrays of [x, y] positions for RMSE calculation.
        """
        if not self.ground_truth_data or not self.ekf_data or not self.noisy_data:
            return None, None, None

        ekf_times = np.array([t for t, p in self.ekf_data])
        noisy_times = np.array([t for t, p in self.noisy_data])

        aligned_gt, aligned_ekf, aligned_noisy = [], [], []

        for gt_time, gt_pose in self.ground_truth_data:
            ekf_idx = np.argmin(np.abs(ekf_times - gt_time))
            noisy_idx = np.argmin(np.abs(noisy_times - gt_time))

            aligned_gt.append([gt_pose.position.x, gt_pose.position.y])
            aligned_ekf.append([self.ekf_data[ekf_idx][1].position.x, self.ekf_data[ekf_idx][1].position.y])
            aligned_noisy.append([self.noisy_data[noisy_idx][1].position.x, self.noisy_data[noisy_idx][1].position.y])

        if not aligned_gt:
            return None, None, None

        return np.array(aligned_gt), np.array(aligned_ekf), np.array(aligned_noisy)

    # --- Core GA Functions ---
    def run_optimizer(self):
        if not self.setup_optimizer_sync():
            self.get_logger().error("Optimizer setup failed. Shutting down.")
            return

        self.get_logger().info("Starting Genetic Algorithm...")
        population = self.initialize_population()
        best_overall_individual = None
        best_overall_fitness = -1

        for generation in range(self.num_generations):
            self.get_logger().info(f"--- Generation {generation + 1}/{self.num_generations} ---")

            fitness_scores = np.array([self.evaluate_fitness(ind) for ind in population])

            best_gen_idx = np.argmax(fitness_scores)
            if fitness_scores[best_gen_idx] > best_overall_fitness:
                best_overall_fitness = fitness_scores[best_gen_idx]
                best_overall_individual = population[best_gen_idx].copy()
                self.get_logger().info(f"!!! New Best Overall Fitness Found: {best_overall_fitness:.4f} !!!")
                self.log_individual(best_overall_individual)
            else:
                self.get_logger().info(f"Best Generation Fitness: {fitness_scores[best_gen_idx]:.4f}")
                self.log_individual(population[best_gen_idx])

            parents = self.selection(population, fitness_scores)
            
            next_population = []
            elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
            for i in elite_indices:
                next_population.append(population[i].copy())

            while len(next_population) < self.population_size:
                parent1, parent2 = parents[np.random.randint(0, len(parents))], parents[np.random.randint(0, len(parents))]
                offspring = self.crossover(parent1, parent2) if np.random.rand() < self.crossover_rate else parent1.copy()
                next_population.append(self.mutate(offspring))

            population = np.array(next_population)

        self.get_logger().info("--- Genetic Algorithm Finished ---")
        self.get_logger().info("Optimal Covariances Found:")
        self.log_individual(best_overall_individual)

    def evaluate_fitness(self, individual):
        self.set_kalman_params(individual)
        time.sleep(0.2) # Reduced sleep

        self.reset_simulation_state()
        self.ground_truth_data.clear()
        self.ekf_data.clear()
        self.noisy_data.clear()
        time.sleep(0.2) # Reduced sleep

        self.execute_defined_path()

        gt, ekf, noisy = self.align_and_get_poses()

        if gt is None:
            self.get_logger().error("Could not align poses, returning fitness of 0.")
            return 0.0
        
        rmse_filtered_vs_truth = np.sqrt(np.mean((gt - ekf)**2))
        rmse_noisy_vs_truth = np.sqrt(np.mean((gt - noisy)**2))
        
        if rmse_filtered_vs_truth < 1e-6:
             rmse_filtered_vs_truth = 1e-6

        accuracy_score = 1.0 / (1.0 + rmse_filtered_vs_truth)
        improvement_ratio = rmse_noisy_vs_truth / rmse_filtered_vs_truth
        fitness =  accuracy_score * improvement_ratio

        self.get_logger().info(
            f"Aligned Points: {len(gt)}, RMSE(filt): {rmse_filtered_vs_truth:.4f}, Fitness: {fitness:.4f}"
        )
        return fitness

    # --- GA Operators ---
    def initialize_population(self):
        return np.random.uniform(low=self.gene_min_val, high=self.gene_max_val, size=(self.population_size, self.num_genes))

    def selection(self, population, fitness_scores):
        selected = []
        for _ in range(self.population_size):
            i1, i2 = np.random.choice(len(population), 2, replace=False)
            winner = i1 if fitness_scores[i1] > fitness_scores[i2] else i2
            selected.append(population[winner])
        return selected

    def crossover(self, parent1, parent2):
        if self.num_genes < 2: return parent1.copy()
        point = np.random.randint(1, self.num_genes)
        return np.concatenate([parent1[:point], parent2[point:]])

    def mutate(self, individual):
        for i in range(self.num_genes):
            if np.random.rand() < self.mutation_rate:
                change = np.random.uniform(-0.5, 0.5) * individual[i] 
                individual[i] = np.clip(individual[i] + change, self.gene_min_val, self.gene_max_val)
        return individual

    # --- Helper Functions ---
    def set_kalman_params(self, individual):
        end_q = self.q_size
        end_odom_r = end_q + self.odom_r_size
        
        params_to_set = [
            Parameter('Q_diag', Parameter.Type.DOUBLE_ARRAY, individual[0:end_q].tolist()),
            Parameter('odom_sensor.R_diag', Parameter.Type.DOUBLE_ARRAY, individual[end_q:end_odom_r].tolist()),
            Parameter('imu_sensor.R_diag', Parameter.Type.DOUBLE_ARRAY, individual[end_odom_r:].tolist())
        ]
        future = self.param_client.set_parameters(params_to_set)
        rclpy.spin_until_future_complete(self, future)

    def reset_simulation_state(self):
        """Resets simulation state and waits for services to complete."""
        gz_req = SetEntityPose.Request()
        gz_req.entity.name = "Sagan"; gz_req.entity.type = Entity.MODEL
        gz_req.pose.position = Point(x=0.0, y=0.0, z=0.2)
        gz_req.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        future_gz = self.reset_gz_pose_client.call_async(gz_req)
        rclpy.spin_until_future_complete(self, future_gz)

        reset_req = Trigger.Request()
        future_odom = self.reset_odom_client.call_async(reset_req)
        rclpy.spin_until_future_complete(self, future_odom)
        future_ekf = self.reset_ekf_client.call_async(reset_req)
        rclpy.spin_until_future_complete(self, future_ekf)

    def execute_defined_path(self):
        path = [(0.6, 0.3, 5.0), (0.6, -0.4, 5.0), (0.6, 0.0, 5.0), (1.2, 0.4, 5.0)]
        for linear, angular, duration in path:
            self.publish_velocity(linear, angular)
            self.spin_for_duration(duration)
        self.publish_velocity(0.0, 0.0)
        time.sleep(1)

    def publish_velocity(self, linear, angular):
        twist = Twist(); twist.linear.x = linear; twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)

    def spin_for_duration(self, duration_sec):
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration_sec:
            rclpy.spin_once(self, timeout_sec=0.01)

    def log_individual(self, individual):
        end_q = self.q_size
        end_odom_r = end_q + self.odom_r_size
        q = np.round(individual[0:end_q], 4).tolist()
        r_o = np.round(individual[end_q:end_odom_r], 4).tolist()
        r_i = np.round(individual[end_odom_r:], 4).tolist()
        self.get_logger().info(f"    Q_diag: {q}")
        self.get_logger().info(f"    odom_sensor.R_diag: {r_o}")
        self.get_logger().info(f"    imu_sensor.R_diag: {r_i}")

def main(args=None):
    rclpy.init(args=args)
    optimizer_node = SaganKalmanFilterOptimizer()
    optimizer_node.run_optimizer()
    optimizer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

