#include "multi_drone_state/manager_node.hpp"

#include <rclcpp_components/register_node_macro.hpp>

namespace manager_node_cpp {

/* ManagerNode() //{ */
ManagerNode::ManagerNode(const rclcpp::NodeOptions &options)
    : rclcpp_lifecycle::LifecycleNode("manager_node", "", options) {
  RCLCPP_INFO(get_logger(), "Creating");

  // Timer execution rate.
  declare_parameter("rate.timer_manager", rclcpp::ParameterValue(100.0));

  // List of all UAV names.
  declare_parameter("uavs_names", std::vector<std::string>{"undefined"});

  // Topic on which neighboring UAVs publish odometry.
  declare_parameter("topic_odom", std::string{"undefined"});

  // Name of the UAV running this node.
  declare_parameter("this_uav_name", std::string{"undefined"});
}

ManagerNode::~ManagerNode() = default;
//}

/* on configure() //{ */
CallbackReturn ManagerNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_DEBUG(get_logger(), "Configuring");

  get_parameters();
  configure_publishers_and_subscriptions();
  configure_timers();
  configure_services();
  configure_clients();

  return CallbackReturn::SUCCESS;
}
//}

/* on activate() //{ */
CallbackReturn ManagerNode::on_activate(
    [[maybe_unused]] const rclcpp_lifecycle::State &state) {
  RCLCPP_DEBUG(get_logger(), "Activating");

  pub_neighbor_position_velocity_->on_activate();
  is_active_ = true;

  return CallbackReturn::SUCCESS;
}
//}

/* on deactivate() //{ */
CallbackReturn ManagerNode::on_deactivate(
    [[maybe_unused]] const rclcpp_lifecycle::State &state) {
  RCLCPP_DEBUG(get_logger(), "Deactivating");

  pub_neighbor_position_velocity_->on_deactivate();
  is_active_ = false;

  return CallbackReturn::SUCCESS;
}
//}

/* on cleanup() //{ */
CallbackReturn
ManagerNode::on_cleanup([[maybe_unused]] const rclcpp_lifecycle::State &state) {
  RCLCPP_DEBUG(get_logger(), "Cleaning up");

  return CallbackReturn::SUCCESS;
}
//}

/* on shutdown() //{ */
CallbackReturn ManagerNode::on_shutdown(
    [[maybe_unused]] const rclcpp_lifecycle::State &state) {
  RCLCPP_DEBUG(get_logger(), "Shutting down");

  is_active_ = false;

  return CallbackReturn::SUCCESS;
}
//}

/* get parameters() //{ */
void ManagerNode::get_parameters() {
  RCLCPP_DEBUG(get_logger(), "Loading parameters");

  get_parameter("rate.timer_manager", timer_manager_rate_);
  get_parameter("topic_odom", odometry_topic_);
  get_parameter("this_uav_name", this_uav_name_);
  get_parameter("uavs_names", uav_names_);
}
//}

/* configure publisher and subscriptions() //{ */
void ManagerNode::configure_publishers_and_subscriptions() {
  RCLCPP_DEBUG(get_logger(), "Configuring publishers and subscriptions");

  neighbors_states_.clear();
  subs_neighbors_position_velocity_.clear();
  is_this_uav_in_neighbors_ = false;

  for (const auto &uav_name : uav_names_) {
    if (uav_name == this_uav_name_) {
      is_this_uav_in_neighbors_ = true;
      continue;
    }

    RCLCPP_INFO(get_logger(), "UAV neighbor: %s", uav_name.c_str());

    const std::string topic_name = "/" + uav_name + "/" + odometry_topic_;

    const std::size_t neighbor_index = neighbors_states_.size();

    neighbors_states_.emplace_back();

    const auto neighbor_subscription =
        create_subscription<nav_msgs::msg::Odometry>(
            topic_name, 1,
            [this,
             neighbor_index](const nav_msgs::msg::Odometry::SharedPtr message) {
              sub_neighbor_odometry(message, neighbor_index);
            });

    subs_neighbors_position_velocity_.push_back(neighbor_subscription);
  }

  if (!is_this_uav_in_neighbors_) {
    RCLCPP_ERROR(get_logger(), "UAV name %s is not in the list of all UAVs.",
                 this_uav_name_.c_str());
  }

  sub_odometry_ = create_subscription<nav_msgs::msg::Odometry>(
      "odometry_in", 1,
      std::bind(&ManagerNode::sub_odometry, this, std::placeholders::_1));

  pub_neighbor_position_velocity_ =
      create_publisher<laser_msgs::msg::NeighborOdomArray>("neighbor_odom_out",
                                                           1);
}
//}

/* configure timers() //{ */
void ManagerNode::configure_timers() {
  RCLCPP_DEBUG(get_logger(), "Configuring timers");

  timer_manager_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / timer_manager_rate_),
      std::bind(&ManagerNode::timer_manager_callback, this), nullptr);
}
//}

/* configure clients() //{ */
void ManagerNode::configure_clients() {
  RCLCPP_DEBUG(get_logger(), "Configuring clients");
}
//}

/* configure services() //{ */
void ManagerNode::configure_services() {
  RCLCPP_DEBUG(get_logger(), "Configuring services");
}
//}

/* sub odometry() //{ */
void ManagerNode::sub_odometry(const nav_msgs::msg::Odometry &message) {
  if (!is_active_) {
    return;
  }

  odometry_ = message;
}
//}

/* timer manager callback() //{ */
void ManagerNode::timer_manager_callback() {
  if (!is_active_) {
    return;
  }

  if (!first_odometry_received_) {
    return;
  }

  {
    std::scoped_lock lock(neighbors_copy_mutex_);
    neighbors_states_aux_ = neighbors_states_;
  }

  std::size_t neighbor_index = 0;

  for (const auto &uav_name : uav_names_) {
    if (uav_name == this_uav_name_) {
      continue;
    }

    const double latency =
        (now() -
         rclcpp::Time(neighbors_states_aux_[neighbor_index].header.stamp))
            .seconds();

    if (latency > 1.0) {
      RCLCPP_WARN(get_logger(),
                  "%s odometry latency is greater than 1 s: %.3f s",
                  uav_name.c_str(), latency);
    }

    ++neighbor_index;
  }

  std::sort(
      neighbors_states_aux_.begin(), neighbors_states_aux_.end(),
      [](const auto &first_neighbor, const auto &second_neighbor) {
        const Eigen::Vector3d first_position(first_neighbor.pose.position.x,
                                             first_neighbor.pose.position.y,
                                             first_neighbor.pose.position.z);

        const Eigen::Vector3d second_position(second_neighbor.pose.position.x,
                                              second_neighbor.pose.position.y,
                                              second_neighbor.pose.position.z);

        return first_position.squaredNorm() < second_position.squaredNorm();
      });

  constexpr std::size_t maximum_neighbors = 5;

  if (neighbors_states_aux_.size() > maximum_neighbors) {
    neighbors_states_aux_.resize(maximum_neighbors);
  }

  neighbor_position_velocity_.array = std::move(neighbors_states_aux_);

  pub_neighbor_position_velocity_->publish(neighbor_position_velocity_);
}
//}

/* sub neighbor odometry() //{ */
void ManagerNode::sub_neighbor_odometry(
    const nav_msgs::msg::Odometry::SharedPtr message,
    const std::size_t neighbor_index) {
  if (!first_odometry_received_) {
    first_odometry_received_ = true;
  }

  neighbors_states_[neighbor_index].child_frame_id = this_uav_name_ + "/fcu";

  neighbors_states_[neighbor_index].header.frame_id = this_uav_name_ + "/fcu";

  neighbors_states_[neighbor_index].header.stamp = message->header.stamp;

  const Eigen::Vector3d this_uav_position(odometry_.pose.pose.position.x,
                                          odometry_.pose.pose.position.y,
                                          odometry_.pose.pose.position.z);

  const Eigen::Vector3d neighbor_position(message->pose.pose.position.x,
                                          message->pose.pose.position.y,
                                          message->pose.pose.position.z);

  const Eigen::Vector3d this_uav_velocity(odometry_.twist.twist.linear.x,
                                          odometry_.twist.twist.linear.y,
                                          odometry_.twist.twist.linear.z);

  const Eigen::Vector3d neighbor_velocity(message->twist.twist.linear.x,
                                          message->twist.twist.linear.y,
                                          message->twist.twist.linear.z);

  const Eigen::Vector3d relative_position =
      neighbor_position - this_uav_position;

  const Eigen::Vector3d relative_velocity =
      this_uav_velocity - neighbor_velocity;

  neighbors_states_[neighbor_index].pose.position.x = relative_position.x();

  neighbors_states_[neighbor_index].pose.position.y = relative_position.y();

  neighbors_states_[neighbor_index].pose.position.z = relative_position.z();

  neighbors_states_[neighbor_index].twist.linear.x = relative_velocity.x();

  neighbors_states_[neighbor_index].twist.linear.y = relative_velocity.y();

  neighbors_states_[neighbor_index].twist.linear.z = relative_velocity.z();
}
//}

} // namespace manager_node_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(manager_node_cpp::ManagerNode)
