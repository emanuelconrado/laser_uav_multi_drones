#ifndef LASER_UAV_MULTI_DRONES__PAYLOAD_TRANSPORT_NODE_HPP
#define LASER_UAV_MULTI_DRONES__PAYLOAD_TRANSPORT_NODE_HPP

#include <Eigen/Dense>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <laser_msgs/srv/compute_ppo_shield.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

namespace laser_uav_multi_drones
{

using CallbackReturn =
  rclcpp_lifecycle::node_interfaces::
  LifecycleNodeInterface::CallbackReturn;

class PayloadTransport
  : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit PayloadTransport(
    const rclcpp::NodeOptions & options =
    rclcpp::NodeOptions());

  ~PayloadTransport() override;

private:
  using ComputePpoShield =
    laser_msgs::srv::ComputePpoShield;

  static constexpr std::size_t
    kObservationSize = 29U;

  static constexpr std::size_t
    kActionSize = 6U;

  static constexpr std::size_t
    kActionsPerUav = 3U;

  // Lifecycle callbacks.
  CallbackReturn on_configure(
    const rclcpp_lifecycle::State &
    previous_state) override;

  CallbackReturn on_activate(
    const rclcpp_lifecycle::State &
    previous_state) override;

  CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State &
    previous_state) override;

  CallbackReturn on_cleanup(
    const rclcpp_lifecycle::State &
    previous_state) override;

  CallbackReturn on_shutdown(
    const rclcpp_lifecycle::State &
    previous_state) override;

  // Node configuration.
  void get_parameters();

  void configure_publishers_and_subscriptions();

  void configure_timers();

  void configure_clients();
  void sub_rope_tension(
    std_msgs::msg::Float64MultiArray::SharedPtr msg);

// Subscription callbacks.

  void sub_uav_odometry(
    nav_msgs::msg::Odometry::SharedPtr msg,
    std::size_t uav_index);

  rclcpp::Subscription<
    std_msgs::msg::Float64MultiArray
  >::SharedPtr sub_rope_tension_;

  void sub_payload_odometry(
    nav_msgs::msg::Odometry::SharedPtr msg);

  // Timer callback.
  void timer_manager_callback();

  // Python PPO/shield client.
  void request_ppo_shield_action();

  // State calculations.
  void updateGlobalObservation();

  Eigen::Vector3d calculateCableTension(
    const Eigen::Vector3d & uav_position,
    const Eigen::Vector3d & uav_velocity,
    const Eigen::Vector3d & payload_position,
    const Eigen::Vector3d & payload_velocity);

  // UAV states. Indices follow uav_names_.
  std::vector<nav_msgs::msg::Odometry>
  uavs_odom_;

  std::vector<nav_msgs::msg::Odometry>
  uavs_odom_aux_;

  std::vector<bool>
  uavs_odometry_received_;

  // Optional conversion from local spawn frame
  // to the global world frame.
  std::vector<Eigen::Vector3d>
  uav_spawn_offsets_;

  // Payload state.
  nav_msgs::msg::Odometry
    payload_odom_;

  nav_msgs::msg::Odometry
    payload_odom_aux_;

  // Publisher.
  rclcpp_lifecycle::LifecyclePublisher<
    std_msgs::msg::Float64MultiArray
  >::SharedPtr pub_action_;

  // UAV subscriptions.
  std::vector<
    rclcpp::Subscription<
      nav_msgs::msg::Odometry
    >::SharedPtr
  > subs_uavs_odometry_;

  // Payload subscription.
  rclcpp::Subscription<
    nav_msgs::msg::Odometry
  >::SharedPtr sub_payload_odometry_;

  // Python PPO/shield service client.
  rclcpp::Client<
    ComputePpoShield
  >::SharedPtr ppo_shield_client_;

  // Timer.
  rclcpp::TimerBase::SharedPtr
    timer_manager_;

  // Parameters.
  double timer_manager_rate_{0.0};

  std::vector<std::string>
  uav_names_;

  std::string this_uav_name_;
  std::string odometry_topic_;

  double cable_length_{0.0};
  double cable_K_{0.0};
  double cable_D_{0.0};

  double ppo_shield_timeout_seconds_{0.14};

  // Policy request state.
  rclcpp::Time policy_start_time_;

  std::atomic_bool
    ppo_shield_request_pending_{false};

  std::atomic<std::uint64_t>
  ppo_shield_sequence_{0U};

  std::chrono::steady_clock::time_point
    ppo_shield_request_start_;

  // Policy input and output.
  std::vector<double>
  global_observation_;

  std::vector<double>
  action_;
  std::vector<double>
  rope_tension_data_;

  std::vector<double>
  rope_tension_data_aux_;

  std::chrono::steady_clock::time_point
    rope_tension_received_time_;

  std::chrono::steady_clock::time_point
    rope_tension_received_time_aux_;

  std::string rope_tension_topic_{
    "/payload/rope_tension"};

  double maximum_rope_tension_age_{
    0.2};

  double virtual_total_tension_{
    0.0};

  double physical_total_tension_{
    0.0};

  bool first_rope_tension_received_{
    false};

// Published action.
  std_msgs::msg::Float64MultiArray
    action_msg_;

  // Synchronization.
  std::mutex odometry_copy_mutex_;

  // Lifecycle and odometry state.
  std::atomic_bool is_active_{false};

  bool first_time{true};
  bool all_uavs_odometry_received_{false};
  bool first_odometry_payload_received_{false};

  double pi{3.14159265359};
};

}  // namespace laser_uav_multi_drones

#endif  // LASER_UAV_MULTI_DRONES__PAYLOAD_TRANSPORT_NODE_HPP
