#ifndef LASER_UAV_MULTI_DRONES__PAYLOAD_TRANSPORT_NODE_HPP
#define LASER_UAV_MULTI_DRONES__PAYLOAD_TRANSPORT_NODE_HPP

#include <Eigen/Dense>
#include <laser_msgs/msg/neighbor_odom.hpp>
#include <laser_msgs/msg/neighbor_odom_array.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

namespace laser_uav_multi_drones
{

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class PayloadTransport : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit PayloadTransport(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  ~PayloadTransport() override;

private:
  // Lifecycle callbacks.
  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;

  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & previous_state) override;

  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & previous_state) override;

  // Node configuration.
  void get_parameters();
  void configure_publishers_and_subscriptions();
  void configure_timers();
  void configure_services();
  void configure_clients();

  // Subscription callbacks.
  void sub_odometry(const nav_msgs::msg::Odometry::SharedPtr msg);
  void sub_payload_odometry(const nav_msgs::msg::Odometry::SharedPtr msg);

  void sub_neighbor_odometry(
    nav_msgs::msg::Odometry::SharedPtr msg, std::size_t neighbor_index);

  // Timer callbacks.
  void timer_manager_callback();

  // Func
  void updateGlobalObservation();

  Eigen::Vector3d calculateCableTension(
    const Eigen::Vector3d & uav_position,
    const Eigen::Vector3d & uav_velocity,
    const Eigen::Vector3d & payload_position,
    const Eigen::Vector3d & payload_velocity);

  // UAV state.
  nav_msgs::msg::Odometry payload_odom_;
  nav_msgs::msg::Odometry payload_odom_aux_;

  nav_msgs::msg::Odometry odometry_;
  nav_msgs::msg::Odometry odometry_aux_;

  std::vector<nav_msgs::msg::Odometry> neighbors_odom_;
  std::vector<nav_msgs::msg::Odometry> neighbors_odom_aux_;

  // Publishers.

  // Subscriptions.
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odometry_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_payload_odometry_;

  std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr>
  subs_neighbors_position_velocity_;

  // Timers.
  rclcpp::TimerBase::SharedPtr timer_manager_;

  // Parameters.
  double timer_manager_rate_{0.0};
  std::vector<std::string> uav_names_;
  std::string this_uav_name_;
  std::string odometry_topic_;
  std::vector<double> global_observation_;

  double cable_length_;
  double cable_K_;
  double cable_D_;

  double pi{3.14159265359};

  // Synchronization.
  std::mutex neighbors_copy_mutex_;

  // Node state.
  bool is_active_{false};
  bool first_time{true};
  bool is_this_uav_in_neighbors_{false};
  bool first_odometry_received_{false};
  bool first_odometry_payload_received_{false};
  bool first_odometry_neighbors_received_{false};
};

}  // namespace laser uav multi drones

#endif  // LASER_UAV_MULTI_DRONES__DRONE_AVOIDANCE_NODE_HPP
