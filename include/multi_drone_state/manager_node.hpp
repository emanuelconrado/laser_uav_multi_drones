#ifndef MANAGER_NODE_CPP__MANAGER_NODE_HPP
#define MANAGER_NODE_CPP__MANAGER_NODE_HPP

#include <Eigen/Dense>
#include <laser_msgs/msg/neighbor_odom.hpp>
#include <laser_msgs/msg/neighbor_odom_array.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

namespace manager_node_cpp
{

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class ManagerNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit ManagerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  ~ManagerNode() override;

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
  void sub_odometry(const nav_msgs::msg::Odometry & message);

  void sub_neighbor_odometry(
    nav_msgs::msg::Odometry::SharedPtr message, std::size_t neighbor_index);

  // Timer callbacks.
  void timer_manager_callback();

  // UAV state.
  nav_msgs::msg::Odometry odometry_;
  std::vector<laser_msgs::msg::NeighborOdom> neighbors_states_;
  std::vector<laser_msgs::msg::NeighborOdom> neighbors_states_aux_;
  laser_msgs::msg::NeighborOdomArray neighbor_position_velocity_;

  // Publishers.
  rclcpp_lifecycle::LifecyclePublisher<laser_msgs::msg::NeighborOdomArray>::SharedPtr
    pub_neighbor_position_velocity_;

  // Subscriptions.
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odometry_;

  std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr>
    subs_neighbors_position_velocity_;

  // Timers.
  rclcpp::TimerBase::SharedPtr timer_manager_;

  // Parameters.
  double timer_manager_rate_{0.0};
  std::vector<std::string> uav_names_;
  std::string this_uav_name_;
  std::string odometry_topic_;

  // Synchronization.
  std::mutex neighbors_copy_mutex_;

  // Node state.
  bool is_active_{false};
  bool is_this_uav_in_neighbors_{false};
  bool first_odometry_received_{false};
};

}  // namespace manager_node_cpp

#endif  // MULTI_DRONE_STATE__MANAGER_NODE_HPP_
