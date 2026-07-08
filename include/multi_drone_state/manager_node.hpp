#ifndef MANAGER_NODE_CPP__MANAGER_NODE_HPP
#define MANAGER_NODE_CPP__MANAGER_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include <Eigen/Dense>

#include <nav_msgs/msg/odometry.hpp>
#include <laser_msgs/msg/neighbor_odom_array.hpp>
#include <laser_msgs/msg/neighbor_odom.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace manager_node_cpp
{

class ManagerNode : public rclcpp_lifecycle::LifecycleNode {
public:
  explicit ManagerNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

  ~ManagerNode() override;

private:
  CallbackReturn on_configure(const rclcpp_lifecycle::State &);

  CallbackReturn on_activate(const rclcpp_lifecycle::State &state);

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &state);

  CallbackReturn on_cleanup(const rclcpp_lifecycle::State &state);

  CallbackReturn on_shutdown(const rclcpp_lifecycle::State &state);

  void getParameters();
  void configPubSub();
  void configTimers();
  void configServices();
  void configClients();

  std::vector<laser_msgs::msg::NeighborOdom> neighbors_states_;
  nav_msgs::msg::Odometry                    odometry_;

  laser_msgs::msg::NeighborOdomArray         neighbor_position_velocity_;
  std::vector<laser_msgs::msg::NeighborOdom> neighbors_states_aux;

  rclcpp_lifecycle::LifecyclePublisher<laser_msgs::msg::NeighborOdomArray>::SharedPtr pub_neighbor_position_velocity_;
  std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr>               subs_neighbors_position_velocity_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::ConstSharedPtr sub_odometry_;
  void                                                          subOdometry(const nav_msgs::msg::Odometry &msg);

  void subNeighborOdom(const nav_msgs::msg::Odometry::SharedPtr msg, int index);

  rclcpp::TimerBase::SharedPtr tmr_manager_;
  void                         tmrManager();

  double                   _rate_tmr_manager_;
  std::vector<std::string> _uavs_names_;
  std::string              _this_uav_name_;
  std::string              _topic_odom_;

  std::mutex mutex_neighbors_copy_;

  bool is_active_{false};
  bool is_this_uav_in_neighbor_{false};
  bool first_odom_received_{false};
};
}  // namespace manager_node_cpp

#endif
