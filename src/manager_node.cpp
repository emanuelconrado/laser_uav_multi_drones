#include "multi_drone_state/manager_node.hpp"

namespace manager_node_cpp
{

/* ManagerNode() //{ */
ManagerNode::ManagerNode(const rclcpp::NodeOptions &options) : rclcpp_lifecycle::LifecycleNode("manager_node", "", options) {
  RCLCPP_INFO(get_logger(), "Creating");

  declare_parameter("rate.timer_manager", rclcpp::ParameterValue(1.0));
  declare_parameter("uavs_names", std::vector<std::string>{"undefined"});
  declare_parameter("topic_odom", std::string{"undefined"});
  declare_parameter("this_uav_name", std::string{"undefined"});
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

  pub_neighbor_position_velocity_->on_activate();
  is_active_ = true;

  return CallbackReturn::SUCCESS;
}
//}

/* OnDeactivate() //{ */
CallbackReturn ManagerNode::on_deactivate([[maybe_unused]] const rclcpp_lifecycle::State &state) {
  RCLCPP_INFO(get_logger(), "Deactivating");

  pub_neighbor_position_velocity_->on_deactivate();

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

    std::cout << "UAV Neighbor: " << uav << std::endl;
    std::string topic_name = "/" + uav + "/" + _topic_odom_;

    int index = neighbors_states_.size();

    neighbors_states_.push_back(laser_msgs::msg::NeighborOdom());

    auto sub = this->create_subscription<nav_msgs::msg::Odometry>(
        topic_name, 1, [this, index](const nav_msgs::msg::Odometry::SharedPtr msg) { this->subNeighborOdom(msg, index); });

    subs_neighbors_position_velocity_.push_back(sub);
  }

  if (!is_this_uav_in_neighbor_) {
    RCLCPP_ERROR(this->get_logger(), "UAV name %s is not in the list of all UAVs.", _this_uav_name_.c_str());
  }

  sub_odometry_ = create_subscription<nav_msgs::msg::Odometry>("odometry_in", 1, std::bind(&ManagerNode::subOdometry, this, std::placeholders::_1));
  pub_neighbor_position_velocity_ = create_publisher<laser_msgs::msg::NeighborOdomArray>("neighbor_odom_out", 1);
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

/* subOdometryGps() //{ */
void ManagerNode::subOdometry(const nav_msgs::msg::Odometry &msg) {
  if (!is_active_) {
    return;
  }

  odometry_ = msg;
}
//}

/* tmrManager() //{ */
void ManagerNode::tmrManager() {
  if (!is_active_) {
    return;
  }

  if (!first_odom_received_) {
    return;
  }

  {
    std::scoped_lock lock(mutex_neighbors_copy_);
    neighbors_states_aux = neighbors_states_;
  }

  int index = 0;

  for (const auto &uav : _uavs_names_) {
    if (uav == _this_uav_name_) {
      is_this_uav_in_neighbor_ = true;
      continue;
    }

    const double latency = (this->now() - rclcpp::Time(neighbors_states_aux[index].header.stamp)).seconds();

    if (latency > 1.0) {
      RCLCPP_WARN(get_logger(), "%s odom latency is greater than 1 s: %.3f s", uav.c_str(), latency);
    }

    ++index;
  }

  std::sort(neighbors_states_aux.begin(), neighbors_states_aux.end(), [](const auto &a, const auto &b) {
    Eigen::Vector3d pos_a(a.pose.position.x, a.pose.position.y, a.pose.position.z);

    Eigen::Vector3d pos_b(b.pose.position.x, b.pose.position.y, b.pose.position.z);

    return pos_a.squaredNorm() < pos_b.squaredNorm();
  });


  if (neighbors_states_aux.size() > 5)
    neighbors_states_aux.resize(5);

  neighbor_position_velocity_.array = std::move(neighbors_states_aux);

  pub_neighbor_position_velocity_->publish(neighbor_position_velocity_);
}
//}

/* SubNeighborOdom() //{ */
void ManagerNode::subNeighborOdom(const nav_msgs::msg::Odometry::SharedPtr msg, int index) {
  if (!first_odom_received_) {
    first_odom_received_ = true;
  }

  neighbors_states_[index].child_frame_id  = _this_uav_name_ + "/fcu";
  neighbors_states_[index].header.frame_id = _this_uav_name_ + "/fcu";
  neighbors_states_[index].header.stamp    = msg->header.stamp;

  Eigen::Vector3d this_uav_position(odometry_.pose.pose.position.x, odometry_.pose.pose.position.y, odometry_.pose.pose.position.z);
  Eigen::Vector3d neighbor_position(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);

  Eigen::Vector3d this_uav_velocity(odometry_.twist.twist.linear.x, odometry_.twist.twist.linear.y, odometry_.twist.twist.linear.z);
  Eigen::Vector3d neighbor_velocity(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);

  Eigen::Vector3d relative_position = neighbor_position - this_uav_position;
  Eigen::Vector3d relative_velocity = this_uav_velocity - neighbor_velocity;

  neighbors_states_[index].pose.position.x = relative_position.x();
  neighbors_states_[index].pose.position.y = relative_position.y();
  neighbors_states_[index].pose.position.z = relative_position.z();

  neighbors_states_[index].twist.linear.x = relative_velocity.x();
  neighbors_states_[index].twist.linear.y = relative_velocity.y();
  neighbors_states_[index].twist.linear.z = relative_velocity.z();
}
//}

}  // namespace manager_node_cpp

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(manager_node_cpp::ManagerNode)
