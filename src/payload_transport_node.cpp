#include <rclcpp_components/register_node_macro.hpp>
#include <laser_uav_multi_drones/payload_transport_node.hpp>


namespace laser_uav_multi_drones
{

/* PayloadTransport() //{ */
PayloadTransport::PayloadTransport(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("manager_node", "", options)
{
  RCLCPP_INFO(get_logger(), "Creating");

  declare_parameter("rate.timer_manager", rclcpp::ParameterValue(100.0));
  declare_parameter("uavs_names", std::vector<std::string>{"undefined"});
  declare_parameter("topic_odom", std::string{"undefined"});
  declare_parameter("this_uav_name", std::string{"undefined"});
  declare_parameter("cable.K", rclcpp::ParameterValue(200.0));
  declare_parameter("cable.D", rclcpp::ParameterValue(60.0));
  declare_parameter("cable.length", rclcpp::ParameterValue(1.0));

  payload_odom_ = nav_msgs::msg::Odometry();
  odometry_ = nav_msgs::msg::Odometry();

  constexpr std::size_t kUavStateSize = 6;
  constexpr std::size_t kPayloadStateSize = 6;
  constexpr std::size_t kTrackingErrorSize = 3;
  constexpr std::size_t kTargetVelocitySize = 3;
  constexpr std::size_t kPayloadAngleSize = 2;
  constexpr std::size_t kAdditionalStateSize = 3;

  const std::size_t observation_size =
    uav_names_.size() * kUavStateSize +
    kPayloadStateSize +
    kTrackingErrorSize +
    kTargetVelocitySize +
    kPayloadAngleSize +
    kAdditionalStateSize;

  global_observation_.resize(observation_size, 0.0);
}

PayloadTransport::~PayloadTransport() = default;
//}

/* on configure() //{ */
CallbackReturn PayloadTransport::on_configure(const rclcpp_lifecycle::State &)
{
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
CallbackReturn PayloadTransport::on_activate([[maybe_unused]] const rclcpp_lifecycle::State & state)
{
  RCLCPP_DEBUG(get_logger(), "Activating");

  is_active_ = true;

  return CallbackReturn::SUCCESS;
}
//}

/* on deactivate() //{ */
CallbackReturn PayloadTransport::on_deactivate(
  [[maybe_unused]] const rclcpp_lifecycle::State & state)
{
  RCLCPP_DEBUG(get_logger(), "Deactivating");

  is_active_ = false;

  return CallbackReturn::SUCCESS;
}
//}

/* on cleanup() //{ */
CallbackReturn PayloadTransport::on_cleanup([[maybe_unused]] const rclcpp_lifecycle::State & state)
{
  RCLCPP_DEBUG(get_logger(), "Cleaning up");

  return CallbackReturn::SUCCESS;
}
//}

/* on shutdown() //{ */
CallbackReturn PayloadTransport::on_shutdown([[maybe_unused]] const rclcpp_lifecycle::State & state)
{
  RCLCPP_DEBUG(get_logger(), "Shutting down");

  is_active_ = false;

  return CallbackReturn::SUCCESS;
}
//}

/* get parameters() //{ */
void PayloadTransport::get_parameters()
{
  RCLCPP_DEBUG(get_logger(), "Loading parameters");

  get_parameter("rate.timer_manager", timer_manager_rate_);
  get_parameter("topic_odom", odometry_topic_);
  get_parameter("this_uav_name", this_uav_name_);
  get_parameter("uavs_names", uav_names_);

  get_parameter("cable.K", cable_K_);
  get_parameter("cable.D", cable_D_);
  get_parameter("cable.length", cable_length_);
}
//}

/* configure publisher and subscriptions() //{ */
void PayloadTransport::configure_publishers_and_subscriptions()
{
  RCLCPP_DEBUG(get_logger(), "Configuring publishers and subscriptions");

  neighbors_odom_.clear();
  subs_neighbors_position_velocity_.clear();
  is_this_uav_in_neighbors_ = false;

  for (const auto & uav_name : uav_names_) {
    if (uav_name == this_uav_name_) {
      is_this_uav_in_neighbors_ = true;
      continue;
    }

    RCLCPP_INFO(get_logger(), "UAV neighbor: %s", uav_name.c_str());

    const std::string topic_name = "/" + uav_name + "/" + odometry_topic_;

    const std::size_t neighbor_index = neighbors_odom_.size();

    neighbors_odom_.emplace_back();

    const auto neighbor_subscription = create_subscription<nav_msgs::msg::Odometry>(
      topic_name, 1, [this, neighbor_index](const nav_msgs::msg::Odometry::SharedPtr message) {
        sub_neighbor_odometry(message, neighbor_index);
      });

    subs_neighbors_position_velocity_.push_back(neighbor_subscription);
  }

  if (!is_this_uav_in_neighbors_) {
    RCLCPP_ERROR(
      get_logger(), "UAV name %s is not in the list of all UAVs.",
      this_uav_name_.c_str());
  }

  sub_odometry_ =
    create_subscription<nav_msgs::msg::Odometry>(
    "odometry_in", 1,
    std::bind(&PayloadTransport::sub_odometry, this, std::placeholders::_1));

  sub_payload_odometry_ =
    create_subscription<nav_msgs::msg::Odometry>(
    "odometry_payload_in", 1,
    std::bind(&PayloadTransport::sub_payload_odometry, this, std::placeholders::_1));
}
//}

/* configure timers() //{ */
void PayloadTransport::configure_timers()
{
  RCLCPP_DEBUG(get_logger(), "Configuring timers");

  timer_manager_ = create_wall_timer(
    std::chrono::duration<double>(
      1.0 / timer_manager_rate_), std::bind(
      &PayloadTransport::timer_manager_callback,
      this), nullptr);
}
//}

/* configure clients() //{ */
void PayloadTransport::configure_clients()
{
  RCLCPP_DEBUG(get_logger(), "Configuring clients");
}
//}

/* configure services() //{ */
void PayloadTransport::configure_services()
{
  RCLCPP_DEBUG(get_logger(), "Configuring services");
}
//}

/* calculate cable tension() //{ */
Eigen::Vector3d PayloadTransport::calculateCableTension(
  const Eigen::Vector3d & uav_position,
  const Eigen::Vector3d & uav_velocity,
  const Eigen::Vector3d & payload_position,
  const Eigen::Vector3d & payload_velocity)
{
  const Eigen::Vector3d delta_pos =
    uav_position - payload_position;

  const double delta_pos_norm =
    delta_pos.norm();

  if (delta_pos_norm <= 1e-6) {
    return Eigen::Vector3d::Zero();
  }

  const Eigen::Vector3d direction =
    delta_pos / delta_pos_norm;

  const double strain =
    std::max(0.0, delta_pos_norm - cable_length_);

  const double axial_velocity =
    (uav_velocity - payload_velocity).dot(direction);

  const double tension =
    std::max(
    0.0,
    cable_K_ * strain +
    cable_D_ * axial_velocity);

  return tension * direction;
}
//}

/* update global observation() //{ */
void PayloadTransport::updateGlobalObservation()
{
  constexpr std::size_t kUavStateSize = 6;
  constexpr std::size_t kPayloadStateSize = 6;
  constexpr std::size_t kTrackingErrorSize = 3;
  constexpr std::size_t kTargetVelocitySize = 3;
  constexpr std::size_t kPayloadAngleSize = 2;

  const std::size_t num_uavs = uav_names_.size();

  // Current UAV observation (Position and Velocity)
  global_observation_[0] =
    odometry_aux_.pose.pose.position.x / 10.0;
  global_observation_[1] =
    odometry_aux_.pose.pose.position.y / 10.0;
  global_observation_[2] =
    odometry_aux_.pose.pose.position.z / 10.0;

  global_observation_[3] =
    odometry_aux_.twist.twist.linear.x / 5.0;
  global_observation_[4] =
    odometry_aux_.twist.twist.linear.y / 5.0;
  global_observation_[5] =
    odometry_aux_.twist.twist.linear.z / 5.0;

  Eigen::Vector3d sum_body_center(
    odometry_aux_.pose.pose.position.x,
    odometry_aux_.pose.pose.position.y,
    odometry_aux_.pose.pose.position.z);

  // Neighbor UAV observations
  for (std::size_t i = 0; i < neighbors_odom_aux_.size(); ++i) {
    const auto & neighbor = neighbors_odom_aux_[i];

    const std::size_t offset =
      kUavStateSize * (i + 1);

    global_observation_[offset + 0] =
      neighbor.pose.pose.position.x / 10.0;
    global_observation_[offset + 1] =
      neighbor.pose.pose.position.y / 10.0;
    global_observation_[offset + 2] =
      neighbor.pose.pose.position.z / 10.0;

    global_observation_[offset + 3] =
      neighbor.twist.twist.linear.x / 5.0;
    global_observation_[offset + 4] =
      neighbor.twist.twist.linear.y / 5.0;
    global_observation_[offset + 5] =
      neighbor.twist.twist.linear.z / 5.0;

    sum_body_center += Eigen::Vector3d(
      neighbor.pose.pose.position.x,
      neighbor.pose.pose.position.y,
      neighbor.pose.pose.position.z);
  }

  // Geometric center of all UAVs
  const Eigen::Vector3d geometric_center =
    sum_body_center / static_cast<double>(num_uavs);

  // Payload observation
  const std::size_t payload_offset =
    num_uavs * kUavStateSize;

  global_observation_[payload_offset + 0] =
    payload_odom_aux_.pose.pose.position.x / 10.0;
  global_observation_[payload_offset + 1] =
    payload_odom_aux_.pose.pose.position.y / 10.0;
  global_observation_[payload_offset + 2] =
    payload_odom_aux_.pose.pose.position.z / 10.0;

  global_observation_[payload_offset + 3] =
    payload_odom_aux_.twist.twist.linear.x / 5.0;
  global_observation_[payload_offset + 4] =
    payload_odom_aux_.twist.twist.linear.y / 5.0;
  global_observation_[payload_offset + 5] =
    payload_odom_aux_.twist.twist.linear.z / 5.0;

  const Eigen::Vector3d target_now = Eigen::Vector3d::Zero();
  const Eigen::Vector3d target_vel = Eigen::Vector3d::Zero();

  // Tracking error
  const std::size_t tracking_offset =
    payload_offset + kPayloadStateSize;

  global_observation_[tracking_offset + 0] =
    (target_now.x() - payload_odom_aux_.pose.pose.position.x) / 5.0;
  global_observation_[tracking_offset + 1] =
    (target_now.y() - payload_odom_aux_.pose.pose.position.y) / 5.0;
  global_observation_[tracking_offset + 2] =
    (target_now.z() - payload_odom_aux_.pose.pose.position.z) / 5.0;

  // Target velocity
  const std::size_t target_velocity_offset =
    tracking_offset + kTrackingErrorSize;

  global_observation_[target_velocity_offset + 0] =
    target_vel.x() / 5.0;
  global_observation_[target_velocity_offset + 1] =
    target_vel.y() / 5.0;
  global_observation_[target_velocity_offset + 2] =
    target_vel.z() / 5.0;

  const Eigen::Vector3d payload_position(
    payload_odom_aux_.pose.pose.position.x,
    payload_odom_aux_.pose.pose.position.y,
    payload_odom_aux_.pose.pose.position.z);

  const Eigen::Vector3d payload_velocity(
    payload_odom_aux_.twist.twist.linear.x,
    payload_odom_aux_.twist.twist.linear.y,
    payload_odom_aux_.twist.twist.linear.z);

  const Eigen::Vector3d payload_relative_position =
    payload_position - geometric_center;

  const double theta_x =
    std::atan2(
    payload_relative_position.x(),
    -payload_relative_position.z());

  const double theta_y =
    std::atan2(
    payload_relative_position.y(),
    -payload_relative_position.z());

  const std::size_t payload_angle_offset =
    target_velocity_offset + kTargetVelocitySize;

  global_observation_[payload_angle_offset + 0] = theta_x;
  global_observation_[payload_angle_offset + 1] = theta_y;

  // Total cable tension
  Eigen::Vector3d total_tension_vector =
    Eigen::Vector3d::Zero();

  const Eigen::Vector3d uav_position(
    odometry_aux_.pose.pose.position.x,
    odometry_aux_.pose.pose.position.y,
    odometry_aux_.pose.pose.position.z);

  const Eigen::Vector3d uav_velocity(
    odometry_aux_.twist.twist.linear.x,
    odometry_aux_.twist.twist.linear.y,
    odometry_aux_.twist.twist.linear.z);

  total_tension_vector += calculateCableTension(
    uav_position,
    uav_velocity,
    payload_position,
    payload_velocity);

  for (const auto & neighbor : neighbors_odom_aux_) {
    const Eigen::Vector3d neighbor_position(
      neighbor.pose.pose.position.x,
      neighbor.pose.pose.position.y,
      neighbor.pose.pose.position.z);

    const Eigen::Vector3d neighbor_velocity(
      neighbor.twist.twist.linear.x,
      neighbor.twist.twist.linear.y,
      neighbor.twist.twist.linear.z);

    total_tension_vector += calculateCableTension(
      neighbor_position,
      neighbor_velocity,
      payload_position,
      payload_velocity);
  }

  const double total_tension =
    total_tension_vector.norm() / 10.0;

  const double wind_strength = 0.0;
  const double time_stamp = now().seconds() / 10.0;

  const std::size_t additional_state_offset =
    payload_angle_offset + kPayloadAngleSize;

  global_observation_[additional_state_offset + 0] =
    total_tension;
  global_observation_[additional_state_offset + 1] =
    wind_strength;
  global_observation_[additional_state_offset + 2] =
    time_stamp;

}
//}

/* timer manager callback() //{ */
void PayloadTransport::timer_manager_callback()
{
  if (!is_active_) {
    return;
  }

  if (!first_odometry_received_ && !first_odometry_payload_received_ &&
    !first_odometry_neighbors_received_)
  {
    return;
    std::cout << "blocked" << std::endl;
  }

  {
    std::scoped_lock lock(neighbors_copy_mutex_);
    neighbors_odom_aux_ = neighbors_odom_;
    payload_odom_aux_ = payload_odom_;
    odometry_aux_ = odometry_;
    updateGlobalObservation();
  }

  std::size_t index = 0;

// Current UAV
  std::cout << "uav_position_x: " << global_observation_[index++] << std::endl;
  std::cout << "uav_position_y: " << global_observation_[index++] << std::endl;
  std::cout << "uav_position_z: " << global_observation_[index++] << std::endl;

  std::cout << "uav_velocity_x: " << global_observation_[index++] << std::endl;
  std::cout << "uav_velocity_y: " << global_observation_[index++] << std::endl;
  std::cout << "uav_velocity_z: " << global_observation_[index++] << std::endl;

// Neighbors
  for (std::size_t i = 0; i < neighbors_odom_aux_.size(); ++i) {
    std::cout << "neighbor_" << i << "_position_x: "
              << global_observation_[index++] << std::endl;
    std::cout << "neighbor_" << i << "_position_y: "
              << global_observation_[index++] << std::endl;
    std::cout << "neighbor_" << i << "_position_z: "
              << global_observation_[index++] << std::endl;

    std::cout << "neighbor_" << i << "_velocity_x: "
              << global_observation_[index++] << std::endl;
    std::cout << "neighbor_" << i << "_velocity_y: "
              << global_observation_[index++] << std::endl;
    std::cout << "neighbor_" << i << "_velocity_z: "
              << global_observation_[index++] << std::endl;
  }

// Payload
  std::cout << "payload_position_x: " << global_observation_[index++] << std::endl;
  std::cout << "payload_position_y: " << global_observation_[index++] << std::endl;
  std::cout << "payload_position_z: " << global_observation_[index++] << std::endl;

  std::cout << "payload_velocity_x: " << global_observation_[index++] << std::endl;
  std::cout << "payload_velocity_y: " << global_observation_[index++] << std::endl;
  std::cout << "payload_velocity_z: " << global_observation_[index++] << std::endl;

// Tracking error
  std::cout << "tracking_error_x: " << global_observation_[index++] << std::endl;
  std::cout << "tracking_error_y: " << global_observation_[index++] << std::endl;
  std::cout << "tracking_error_z: " << global_observation_[index++] << std::endl;

// Target velocity
  std::cout << "target_velocity_x: " << global_observation_[index++] << std::endl;
  std::cout << "target_velocity_y: " << global_observation_[index++] << std::endl;
  std::cout << "target_velocity_z: " << global_observation_[index++] << std::endl;

// Payload swing
  std::cout << "theta_x: " << global_observation_[index++] << std::endl;
  std::cout << "theta_y: " << global_observation_[index++] << std::endl;

// Additional observations
  std::cout << "total_tension: " << global_observation_[index++] << std::endl;
  std::cout << "wind_strength: " << global_observation_[index++] << std::endl;
  std::cout << "time_stamp: " << global_observation_[index++] << std::endl;

  const double latency =
    (now() - rclcpp::Time(payload_odom_.header.stamp)).seconds();

  if (latency > 1.0) {
    RCLCPP_WARN(
      get_logger(),
      "payload odometry latency is greater than 1 s: %.3f s",
      latency);
  }
}
//}

/* sub odometry() //{ */
void PayloadTransport::sub_odometry(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  if (!is_active_) {
    return;
  }

  if (!first_odometry_received_) {
    first_odometry_received_ = true;
  }

  odometry_ = *msg;
}
//}

/* sub payload odometry() //{ */
void PayloadTransport::sub_payload_odometry(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  if (!first_odometry_received_) {
    first_odometry_payload_received_ = true;
  }

  payload_odom_ = *msg;
}
//}

/* sub neighbor odometry() //{ */
void PayloadTransport::sub_neighbor_odometry(
  const nav_msgs::msg::Odometry::SharedPtr msg,
  const std::size_t neighbor_index)
{
  if (!first_odometry_received_) {
    first_odometry_neighbors_received_ = true;
  }

  neighbors_odom_[neighbor_index] = *msg;
}
//}

}  // namespace laser uav multi drones

RCLCPP_COMPONENTS_REGISTER_NODE(laser_uav_multi_drones::PayloadTransport)
