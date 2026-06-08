#include "multi_drone_state/manager_node.hpp"

namespace manager_node_cpp
{

/* ManagerNode() //{ */
ManagerNode::ManagerNode(const rclcpp::NodeOptions &options) : rclcpp_lifecycle::LifecycleNode("manager_node", "", options) {
  RCLCPP_INFO(get_logger(), "Creating");

  declare_parameter("rate.timer_manager", rclcpp::ParameterValue(1.0));
  declare_parameter("uavs_names", std::vector<std::string>{"uav1"});
  declare_parameter("topic_odom", std::string{"ground_truth"});
  declare_parameter("this_uav_name", std::string{"uav1"});
}
//}

/* ~ManagerNode() //{ */
ManagerNode::~ManagerNode() {
}
//}

/* OnConfigure() //{ */
CallbackReturn ManagerNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "Configuring");

  getParameters();
  configPubSub();
  configTimers();
  configServices();
  configClients();

  return CallbackReturn::SUCCESS;
}
//}

/* OnActivate() //{ */
CallbackReturn ManagerNode::on_activate([[maybe_unused]] const rclcpp_lifecycle::State &state) {
  RCLCPP_INFO(get_logger(), "Activating");

  pub_neighbor_odom_->on_activate();
  is_active_ = true;

  return CallbackReturn::SUCCESS;
}
//}

/* OnDeactivate() //{ */
CallbackReturn ManagerNode::on_deactivate([[maybe_unused]] const rclcpp_lifecycle::State &state) {
  RCLCPP_INFO(get_logger(), "Deactivating");

  pub_neighbor_odom_->on_deactivate();

  return CallbackReturn::SUCCESS;
}
//}

/* OnClanup() //{ */
CallbackReturn ManagerNode::on_cleanup([[maybe_unused]] const rclcpp_lifecycle::State &state) {
  RCLCPP_INFO(get_logger(), "Cleaning up");

  return CallbackReturn::SUCCESS;
}
//}

/* OnShutdown() //{ */
CallbackReturn ManagerNode::on_shutdown([[maybe_unused]] const rclcpp_lifecycle::State &state) {
  RCLCPP_INFO(get_logger(), "Shutting down");

  return CallbackReturn::SUCCESS;
}
//}

/* GetParameters() //{ */
void ManagerNode::getParameters() {
  RCLCPP_INFO(get_logger(), "Loading parameters");

  get_parameter("rate.timer_manager", _rate_tmr_manager_);
  get_parameter("topic_odom", _topic_odom_);
  get_parameter("this_uav_name", _this_uav_name_);
  get_parameter("uavs_names", _uavs_names_);
}
//}

/* ConfigPubSub() //{ */
void ManagerNode::configPubSub() {
  RCLCPP_INFO(get_logger(), "initPubSub");

  neighbors_states_.clear();

  for (const auto &uav : _uavs_names_) {
    if (uav == _this_uav_name_) {
      is_this_uav_in_neighbor_ = true;
      continue;
    }

    std::cout << "UAV: " << uav << std::endl;
    std::string topic_name = "/" + uav + "/" + _topic_odom_;

    int index = neighbors_states_.size();

    neighbors_states_.push_back(laser_msgs::msg::NeighborOdom());

    auto sub = this->create_subscription<nav_msgs::msg::Odometry>(
        topic_name, 1, [this, index](const nav_msgs::msg::Odometry::SharedPtr msg) { this->subNeighborOdom(msg, index); });

    subs_neighbors_odom_.push_back(sub);
  }

  if (!is_this_uav_in_neighbor_) {
    RCLCPP_ERROR(this->get_logger(), "UAV name %s is not in the list of all UAVs.", _this_uav_name_.c_str());
  }

  pub_neighbor_odom_ = create_publisher<laser_msgs::msg::NeighborOdomArray>("neighbor_odom_out", 1);
}
//}

/* ConfigTimers() //{ */
void ManagerNode::configTimers() {
  RCLCPP_INFO(get_logger(), "initTimers");

  tmr_manager_ = create_wall_timer(std::chrono::duration<double>(1.0 / _rate_tmr_manager_), std::bind(&ManagerNode::tmrManager, this), nullptr);
}
//}

/* ConfigClients() //{ */
void ManagerNode::configClients() {
  RCLCPP_INFO(get_logger(), "initClients");
}
//}

/* ConfigServices() //{ */
void ManagerNode::configServices() {
  RCLCPP_INFO(get_logger(), "initServices");
}
//}

/* tmrManager() //{ */
void ManagerNode::tmrManager() {
  if (!is_active_) {
    return;
  }

  if (first_odom_received_)
    pub_neighbor_odom_->publish(neighbor_odom_);
}
//}

/* SubNeighborOdom() //{ */
void ManagerNode::subNeighborOdom(const nav_msgs::msg::Odometry::SharedPtr msg, int index) {
  if (!first_odom_received_) {
    first_odom_received_ = true;
  }

  neighbors_states_[index].pose = msg->pose.pose;
  neighbors_states_[index].twist    = msg->twist.twist;


  neighbor_odom_.array = neighbors_states_;
}
//}

}  // namespace manager_node_cpp

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(manager_node_cpp::ManagerNode)
