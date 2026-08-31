#include <laser_uav_multi_drones/payload_transport_node.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>

#include <rclcpp_components/register_node_macro.hpp>

namespace laser_uav_multi_drones
{

/* PayloadTransport() //{ */
PayloadTransport::PayloadTransport(
  const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode(
    "payload_transport",
    "",
    options)
{
  RCLCPP_INFO(
    get_logger(),
    "Creating");

  declare_parameter(
    "rate.timer_manager",
    rclcpp::ParameterValue(100.0));

  declare_parameter(
    "uavs_names",
    std::vector<std::string>{"undefined"});

  declare_parameter(
    "topic_odom",
    std::string{"undefined"});

  declare_parameter(
    "this_uav_name",
    std::string{"undefined"});

  declare_parameter(
    "cable.K",
    rclcpp::ParameterValue(200.0));

  declare_parameter(
    "cable.D",
    rclcpp::ParameterValue(60.0));

  declare_parameter(
    "cable.length",
    rclcpp::ParameterValue(1.0));

  declare_parameter(
    "cable.tension_topic",
    std::string{"/payload/rope_tension"});

  declare_parameter(
    "cable.maximum_tension_age_seconds",
    rclcpp::ParameterValue(0.2));

  declare_parameter(
    "ppo_shield.timeout_seconds",
    rclcpp::ParameterValue(0.14));

  global_observation_.assign(
    kObservationSize,
    0.0);

  action_.assign(
    kActionSize,
    0.0);

  action_msg_.data.assign(
    kActionSize,
    0.0);

  payload_odom_ =
    nav_msgs::msg::Odometry{};
}

PayloadTransport::~PayloadTransport() = default;
//}

/* on configure() //{ */
CallbackReturn PayloadTransport::on_configure(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_DEBUG(
    get_logger(),
    "Configuring");

  get_parameters();

  if (uav_names_.size() != 2U) {
    RCLCPP_ERROR(
      get_logger(),
      "The Python PPO service expects exactly "
      "two UAVs, but %zu were configured.",
      uav_names_.size());

    return CallbackReturn::FAILURE;
  }

  if (timer_manager_rate_ <= 0.0) {
    RCLCPP_ERROR(
      get_logger(),
      "rate.timer_manager must be greater than zero.");

    return CallbackReturn::FAILURE;
  }

  if (maximum_rope_tension_age_ <= 0.0) {
    RCLCPP_ERROR(
      get_logger(),
      "cable.maximum_tension_age_seconds must "
      "be greater than zero.");

    return CallbackReturn::FAILURE;
  }

  if (ppo_shield_timeout_seconds_ <= 0.0) {
    RCLCPP_ERROR(
      get_logger(),
      "ppo_shield.timeout_seconds must be "
      "greater than zero.");

    return CallbackReturn::FAILURE;
  }

  // Ground-truth odometries are already in the
  // global world frame.
  uav_spawn_offsets_.assign(
    uav_names_.size(),
    Eigen::Vector3d::Zero());

  first_time = true;

  ppo_shield_sequence_ = 0U;

  ppo_shield_request_pending_.store(
    false);

  {
    std::lock_guard<std::mutex> lock(
      odometry_copy_mutex_);

    rope_tension_data_.clear();
    rope_tension_data_aux_.clear();

    rope_tension_received_time_ =
      std::chrono::steady_clock::time_point{};

    rope_tension_received_time_aux_ =
      std::chrono::steady_clock::time_point{};

    first_rope_tension_received_ =
      false;
  }

  virtual_total_tension_ = 0.0;
  physical_total_tension_ = 0.0;

  configure_publishers_and_subscriptions();
  configure_clients();
  configure_timers();

  return CallbackReturn::SUCCESS;
}
//}

/* on activate() //{ */
CallbackReturn PayloadTransport::on_activate(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_DEBUG(
    get_logger(),
    "Activating");

  pub_action_->on_activate();
  is_active_ = true;

  return CallbackReturn::SUCCESS;
}
//}

/* on deactivate() //{ */
CallbackReturn PayloadTransport::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_DEBUG(
    get_logger(),
    "Deactivating");

  is_active_ = false;

  ppo_shield_request_pending_.store(
    false);

  if (pub_action_) {
    pub_action_->on_deactivate();
  }

  return CallbackReturn::SUCCESS;
}
//}

/* on cleanup() //{ */
CallbackReturn PayloadTransport::on_cleanup(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_DEBUG(
    get_logger(),
    "Cleaning up");

  is_active_ = false;

  ppo_shield_request_pending_.store(
    false);

  timer_manager_.reset();
  ppo_shield_client_.reset();

  subs_uavs_odometry_.clear();
  sub_payload_odometry_.reset();
  sub_rope_tension_.reset();
  pub_action_.reset();

  uavs_odom_.clear();
  uavs_odom_aux_.clear();
  uavs_odometry_received_.clear();

  rope_tension_data_.clear();
  rope_tension_data_aux_.clear();

  rope_tension_received_time_ =
    std::chrono::steady_clock::time_point{};

  rope_tension_received_time_aux_ =
    std::chrono::steady_clock::time_point{};

  virtual_total_tension_ = 0.0;
  physical_total_tension_ = 0.0;

  all_uavs_odometry_received_ = false;
  first_odometry_payload_received_ = false;
  first_rope_tension_received_ = false;
  first_time = true;

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

  get_parameter(
    "cable.tension_topic",
    rope_tension_topic_);

  get_parameter(
    "cable.maximum_tension_age_seconds",
    maximum_rope_tension_age_);

  get_parameter(
    "ppo_shield.timeout_seconds",
    ppo_shield_timeout_seconds_);

  action_.assign(
    uav_names_.size() *
    kActionsPerUav,
    0.0);
}
//}

/* configure publisher and subscriptions() //{ */
void PayloadTransport::configure_publishers_and_subscriptions()
{
  RCLCPP_DEBUG(
    get_logger(),
    "Configuring publishers and subscriptions");

  subs_uavs_odometry_.clear();

  const std::size_t num_uavs =
    uav_names_.size();

  uavs_odom_.assign(
    num_uavs,
    nav_msgs::msg::Odometry{});

  uavs_odom_aux_.assign(
    num_uavs,
    nav_msgs::msg::Odometry{});

  uavs_odometry_received_.assign(
    num_uavs,
    false);

  all_uavs_odometry_received_ = false;

  subs_uavs_odometry_.reserve(
    num_uavs);

  for (std::size_t uav_index = 0;
    uav_index < num_uavs;
    ++uav_index)
  {
    const std::string & uav_name =
      uav_names_[uav_index];

    const std::string topic_name =
      "/" + uav_name + "/" + odometry_topic_;

    RCLCPP_INFO(
      get_logger(),
      "UAV %zu (%s) odometry topic: %s",
      uav_index,
      uav_name.c_str(),
      topic_name.c_str());

    auto subscription =
      create_subscription<nav_msgs::msg::Odometry>(
      topic_name,
      1,
      [this, uav_index](
        const nav_msgs::msg::Odometry::SharedPtr msg)
      {
        sub_uav_odometry(
          msg,
          uav_index);
      });

    subs_uavs_odometry_.push_back(
      subscription);
  }

  const auto this_uav_iterator =
    std::find(
    uav_names_.begin(),
    uav_names_.end(),
    this_uav_name_);

  if (this_uav_iterator == uav_names_.end()) {
    RCLCPP_ERROR(
      get_logger(),
      "UAV name '%s' is not present in uavs_names.",
      this_uav_name_.c_str());
  }

  sub_payload_odometry_ =
    create_subscription<nav_msgs::msg::Odometry>(
    "odometry_payload_in",
    1,
    std::bind(
      &PayloadTransport::sub_payload_odometry,
      this,
      std::placeholders::_1));

  pub_action_ =
    create_publisher<std_msgs::msg::Float64MultiArray>(
    "action_out",
    1);

  sub_rope_tension_ =
    create_subscription<
    std_msgs::msg::Float64MultiArray>(
    rope_tension_topic_,
    rclcpp::QoS(10),
    std::bind(
      &PayloadTransport::sub_rope_tension,
      this,
      std::placeholders::_1));

  RCLCPP_INFO(
    get_logger(),
    "Physical rope tension topic: %s",
    rope_tension_topic_.c_str());
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
  RCLCPP_DEBUG(
    get_logger(),
    "Configuring clients");

  ppo_shield_client_ =
    create_client<ComputePpoShield>(
    "/compute_ppo_shield");

  RCLCPP_INFO(
    get_logger(),
    "PPO Python service client configured: "
    "/compute_ppo_shield");
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
  constexpr std::size_t kUavStateSize = 6U;
  constexpr std::size_t kPayloadStateSize = 6U;
  constexpr std::size_t kTrackingErrorSize = 3U;
  constexpr std::size_t kTargetVelocitySize = 3U;
  constexpr std::size_t kPayloadAngleSize = 2U;
  constexpr std::size_t kSharedStateSize = 17U;

  const std::size_t num_uavs =
    uav_names_.size();

  if (num_uavs == 0U) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "The UAV list is empty.");

    return;
  }

  if (uavs_odom_aux_.size() != num_uavs) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Invalid UAV odometry vector size: "
      "expected %zu, received %zu.",
      num_uavs,
      uavs_odom_aux_.size());

    return;
  }

  if (uav_spawn_offsets_.size() != num_uavs) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Invalid UAV spawn offset vector size: "
      "expected %zu, received %zu.",
      num_uavs,
      uav_spawn_offsets_.size());

    return;
  }

  const std::size_t observation_size =
    kUavStateSize * num_uavs +
    kSharedStateSize;

  if (global_observation_.size() !=
    observation_size)
  {
    global_observation_.assign(
      observation_size,
      0.0);
  }

  std::vector<Eigen::Vector3d>
  uav_global_positions(
    num_uavs,
    Eigen::Vector3d::Zero());

  std::vector<Eigen::Vector3d>
  uav_global_velocities(
    num_uavs,
    Eigen::Vector3d::Zero());

  Eigen::Vector3d geometric_center =
    Eigen::Vector3d::Zero();

  // UAV states in exactly the same order as uav_names_.
  for (std::size_t uav_index = 0U;
    uav_index < num_uavs;
    ++uav_index)
  {
    const auto & uav_odom =
      uavs_odom_aux_[uav_index];

    const Eigen::Vector3d local_position(
      uav_odom.pose.pose.position.x,
      uav_odom.pose.pose.position.y,
      uav_odom.pose.pose.position.z);

    uav_global_positions[uav_index] =
      local_position +
      uav_spawn_offsets_[uav_index];

    uav_global_velocities[uav_index] =
      Eigen::Vector3d(
      uav_odom.twist.twist.linear.x,
      uav_odom.twist.twist.linear.y,
      uav_odom.twist.twist.linear.z);

    const Eigen::Vector3d & global_position =
      uav_global_positions[uav_index];

    const Eigen::Vector3d & global_velocity =
      uav_global_velocities[uav_index];

    const std::size_t offset =
      kUavStateSize *
      uav_index;

    global_observation_[offset + 0U] =
      global_position.x() / 10.0;

    global_observation_[offset + 1U] =
      global_position.y() / 10.0;

    global_observation_[offset + 2U] =
      global_position.z() / 10.0;

    global_observation_[offset + 3U] =
      global_velocity.x() / 5.0;

    global_observation_[offset + 4U] =
      global_velocity.y() / 5.0;

    global_observation_[offset + 5U] =
      global_velocity.z() / 5.0;

    geometric_center +=
      global_position;
  }

  geometric_center /=
    static_cast<double>(num_uavs);

  // This assumes that payload odometry is already in the global frame.
  const Eigen::Vector3d payload_position(
    payload_odom_aux_.pose.pose.position.x,
    payload_odom_aux_.pose.pose.position.y,
    payload_odom_aux_.pose.pose.position.z);

  const Eigen::Vector3d payload_velocity(
    payload_odom_aux_.twist.twist.linear.x,
    payload_odom_aux_.twist.twist.linear.y,
    payload_odom_aux_.twist.twist.linear.z);

  // Payload state.
  const std::size_t payload_offset =
    num_uavs *
    kUavStateSize;

  global_observation_[payload_offset + 0U] =
    payload_position.x() / 10.0;

  global_observation_[payload_offset + 1U] =
    payload_position.y() / 10.0;

  global_observation_[payload_offset + 2U] =
    payload_position.z() / 10.0;

  global_observation_[payload_offset + 3U] =
    payload_velocity.x() / 5.0;

  global_observation_[payload_offset + 4U] =
    payload_velocity.y() / 5.0;

  global_observation_[payload_offset + 5U] =
    payload_velocity.z() / 5.0;

  const Eigen::Vector3d target_now(
    0.0,
    0.0,
    4.0);

  const Eigen::Vector3d target_velocity =
    Eigen::Vector3d::Zero();

  // Tracking error.
  const std::size_t tracking_offset =
    payload_offset +
    kPayloadStateSize;

  const Eigen::Vector3d tracking_error =
    target_now -
    payload_position;

  global_observation_[tracking_offset + 0U] =
    tracking_error.x() / 5.0;

  global_observation_[tracking_offset + 1U] =
    tracking_error.y() / 5.0;

  global_observation_[tracking_offset + 2U] =
    tracking_error.z() / 5.0;

  // Target velocity.
  const std::size_t target_velocity_offset =
    tracking_offset +
    kTrackingErrorSize;

  global_observation_[target_velocity_offset + 0U] =
    target_velocity.x() / 5.0;

  global_observation_[target_velocity_offset + 1U] =
    target_velocity.y() / 5.0;

  global_observation_[target_velocity_offset + 2U] =
    target_velocity.z() / 5.0;

  // Payload swing angles.
  const Eigen::Vector3d payload_relative_position =
    payload_position -
    geometric_center;

  const double theta_x =
    std::atan2(
    payload_relative_position.x(),
    -payload_relative_position.z());

  const double theta_y =
    std::atan2(
    payload_relative_position.y(),
    -payload_relative_position.z());

  const std::size_t payload_angle_offset =
    target_velocity_offset +
    kTargetVelocitySize;

  global_observation_[payload_angle_offset + 0U] =
    theta_x / pi;

  global_observation_[payload_angle_offset + 1U] =
    theta_y / pi;

  // Total cable tension.
  Eigen::Vector3d total_tension_vector =
    Eigen::Vector3d::Zero();

  for (std::size_t uav_index = 0U;
    uav_index < num_uavs;
    ++uav_index)
  {
    total_tension_vector +=
      calculateCableTension(
      uav_global_positions[uav_index],
      uav_global_velocities[uav_index],
      payload_position,
      payload_velocity);
  }

  /*
   * Keep the center-to-center result only for comparison.
   * The PPO observation uses the resultant force published
   * by the Gazebo rope plugin.
   */
  virtual_total_tension_ =
    total_tension_vector.norm();

  if (rope_tension_data_aux_.size() != 14U) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Physical rope tension data is unavailable.");

    return;
  }

  physical_total_tension_ =
    rope_tension_data_aux_[13U];

  const double wind_strength =
    0.0;

  const double policy_elapsed_time =
    std::max(
    0.0,
    (now() - policy_start_time_).seconds());

  const std::size_t additional_state_offset =
    payload_angle_offset +
    kPayloadAngleSize;

  global_observation_[additional_state_offset + 0U] =
    physical_total_tension_ / 20.0;

  global_observation_[additional_state_offset + 1U] =
    wind_strength;

  global_observation_[additional_state_offset + 2U] =
    policy_elapsed_time / 10.0;
}
//}

/* configure request ppo() //{ */
void PayloadTransport::request_ppo_shield_action()
{
  if (!ppo_shield_client_) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "PPO Python service client is not configured.");

    return;
  }

  if (!ppo_shield_client_->service_is_ready()) {
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Waiting for /compute_ppo_shield.");

    return;
  }

  if (global_observation_.size() !=
    kObservationSize)
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Invalid PPO observation size: "
      "expected %zu, received %zu.",
      kObservationSize,
      global_observation_.size());

    return;
  }

  if (ppo_shield_request_pending_.load()) {
    const double elapsed_seconds =
      std::chrono::duration<double>(
      std::chrono::steady_clock::now() -
      ppo_shield_request_start_).count();

    if (elapsed_seconds >
      ppo_shield_timeout_seconds_)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "PPO Python request still pending: "
        "%.3f ms.",
        elapsed_seconds * 1000.0);
    }

    // Do not create a request backlog.
    return;
  }

  auto request =
    std::make_shared<
    ComputePpoShield::Request>();

  request->stamp =
    static_cast<builtin_interfaces::msg::Time>(
    now());

  request->sequence =
    ++ppo_shield_sequence_;

  std::copy(
    global_observation_.begin(),
    global_observation_.end(),
    request->observation.begin());

  const std::uint64_t request_sequence =
    request->sequence;

  const auto request_start =
    std::chrono::steady_clock::now();

  ppo_shield_request_start_ =
    request_start;

  ppo_shield_request_pending_.store(
    true);

  try {
    ppo_shield_client_->async_send_request(
      request,
      [this, request_sequence, request_start](
        rclcpp::Client<
          ComputePpoShield
        >::SharedFuture future)
      {
        ppo_shield_request_pending_.store(
          false);

        const double round_trip_ms =
        std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() -
          request_start).count();

        if (!is_active_) {
          return;
        }

        try {
          const auto response =
          future.get();

          if (response->sequence !=
          request_sequence)
          {
            RCLCPP_WARN(
              get_logger(),
              "Discarding PPO response with "
              "unexpected sequence: expected %lu, "
              "received %lu.",
              static_cast<unsigned long>(
                request_sequence),
              static_cast<unsigned long>(
                response->sequence));

            return;
          }

          if (!response->success) {
            RCLCPP_ERROR(
              get_logger(),
              "PPO Python service failed: %s",
              response->status.c_str());

            return;
          }

          const bool raw_action_is_finite =
          std::all_of(
            response->raw_action.begin(),
            response->raw_action.end(),
            [](double value)
            {
              return std::isfinite(value);
            });

          const bool safe_action_is_finite =
          std::all_of(
            response->safe_action.begin(),
            response->safe_action.end(),
            [](double value)
            {
              return std::isfinite(value);
            });

          if (!raw_action_is_finite ||
          !safe_action_is_finite)
          {
            RCLCPP_ERROR(
              get_logger(),
              "Python PPO response contains "
              "NaN or Inf.");

            return;
          }

          std::cout
            << std::fixed
            << std::setprecision(6)
            << "\n"
            << "================================="
            << "=================================\n"
            << "          PYTHON PPO/SHIELD RESPONSE\n"
            << "================================="
            << "=================================\n"
            << "Sequence: "
            << response->sequence
            << "\n"
            << "Critic value: "
            << response->critic_value
            << "\n"
            << "Python computation: "
            << response->computation_time_ms
            << " ms\n"
            << "ROS round trip: "
            << round_trip_ms
            << " ms\n"
            << "Shield applied: "
            << std::boolalpha
            << response->shield_applied
            << "\n"
            << "Shield intervened: "
            << response->shield_intervened
            << "\n"
            << "Status: "
            << response->status
            << "\n"
            << std::noboolalpha
            << "---------------------------------"
            << "---------------------------------\n"
            << std::setw(8) << "UAV"
            << std::setw(12) << "Raw Fx"
            << std::setw(12) << "Raw Fy"
            << std::setw(12) << "Raw Fz"
            << std::setw(12) << "Safe Fx"
            << std::setw(12) << "Safe Fy"
            << std::setw(12) << "Safe Fz"
            << "\n";

          const std::size_t number_of_actions =
          kActionSize /
          kActionsPerUav;

          for (std::size_t uav_index = 0;
          uav_index < number_of_actions;
          ++uav_index)
          {
            const std::size_t offset =
            uav_index *
            kActionsPerUav;

            const std::string uav_label =
            uav_index < uav_names_.size() ?
            uav_names_[uav_index] :
            "uav" +
            std::to_string(uav_index + 1U);

            std::cout
              << std::setw(8)
              << uav_label
              << std::setw(12)
              << response->raw_action[offset + 0U]
              << std::setw(12)
              << response->raw_action[offset + 1U]
              << std::setw(12)
              << response->raw_action[offset + 2U]
              << std::setw(12)
              << response->safe_action[offset + 0U]
              << std::setw(12)
              << response->safe_action[offset + 1U]
              << std::setw(12)
              << response->safe_action[offset + 2U]
              << "\n";
          }

          std::cout
            << "================================="
            << "=================================\n"
            << std::defaultfloat
            << std::flush;

          if (!response->shield_applied) {
            RCLCPP_WARN_THROTTLE(
              get_logger(),
              *get_clock(),
              2000,
              "PPO response received, but the "
              "Python shield is not active. "
              "Action will not be published.");

            return;
          }

          if (round_trip_ms >
          ppo_shield_timeout_seconds_ *
          1000.0)
          {
            RCLCPP_WARN(
              get_logger(),
              "Discarding stale PPO response: "
              "%.3f ms.",
              round_trip_ms);

            return;
          }

          action_.assign(
            response->safe_action.begin(),
            response->safe_action.end());

          action_msg_.data =
          action_;

          if (pub_action_ &&
          pub_action_->is_activated())
          {
            pub_action_->publish(
              action_msg_);
          }
        } catch (
          const std::exception & error)
        {
          RCLCPP_ERROR(
            get_logger(),
            "Could not process Python PPO "
            "response: %s",
            error.what());
        }
      });
  } catch (const std::exception & error) {
    ppo_shield_request_pending_.store(
      false);

    RCLCPP_ERROR(
      get_logger(),
      "Could not send PPO request: %s",
      error.what());
  }
}
//}

/* timer manager callback() //{ */
void PayloadTransport::timer_manager_callback()
{
  if (!is_active_) {
    return;
  }

  // Copy the latest odometries under the mutex.
  {
    std::lock_guard<std::mutex> lock(
      odometry_copy_mutex_);

    if (
      !all_uavs_odometry_received_ ||
      !first_odometry_payload_received_ ||
      !first_rope_tension_received_)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Waiting for all UAV odometries, payload "
        "odometry and physical rope tension.");

      return;
    }

    uavs_odom_aux_ =
      uavs_odom_;

    payload_odom_aux_ =
      payload_odom_;

    rope_tension_data_aux_ =
      rope_tension_data_;

    rope_tension_received_time_aux_ =
      rope_tension_received_time_;
  }

  const double rope_tension_age =
    std::chrono::duration<double>(
    std::chrono::steady_clock::now() -
    rope_tension_received_time_aux_
    ).count();

  if (rope_tension_age >
    maximum_rope_tension_age_)
  {
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Physical rope tension is stale: %.3f s.",
      rope_tension_age);

    return;
  }

  if (rope_tension_data_aux_.size() != 14U) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Invalid copied rope tension data size: %zu.",
      rope_tension_data_aux_.size());

    return;
  }

  const std::size_t num_uavs =
    uav_names_.size();

  if (uavs_odom_aux_.size() !=
    num_uavs)
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Invalid UAV odometry vector size: "
      "expected %zu, received %zu.",
      num_uavs,
      uavs_odom_aux_.size());

    return;
  }

  const std::size_t expected_observation_size =
    6U * num_uavs + 17U;

  const std::size_t expected_action_size =
    kActionsPerUav * num_uavs;

  // The Python policy currently supports the
  // centralized two-UAV model.
  if (expected_observation_size !=
    kObservationSize ||
    expected_action_size !=
    kActionSize)
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "The Python PPO service expects "
      "observation=%zu and action=%zu, but "
      "the current configuration produces "
      "observation=%zu and action=%zu.",
      kObservationSize,
      kActionSize,
      expected_observation_size,
      expected_action_size);

    return;
  }

  // Wait for stable hover only before the first
  // policy request. After starting, do not suspend
  // control when the state deviates from hover.
  if (first_time) {
    const double payload_z =
      payload_odom_aux_.pose.pose.position.z;

    const double payload_vz =
      payload_odom_aux_.twist.twist.linear.z;

    const bool payload_at_hover =
      std::abs(payload_z - 4.0) <= 0.10 &&
      std::abs(payload_vz) <= 0.05;

    bool uavs_at_hover = true;

    for (const auto & uav_odom :
      uavs_odom_aux_)
    {
      if (std::abs(
          uav_odom.twist.twist.linear.z) >
        0.05)
      {
        uavs_at_hover = false;
        break;
      }
    }

    if (!payload_at_hover ||
      !uavs_at_hover)
    {
      RCLCPP_INFO_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Waiting for stable post-takeoff "
        "hover. Payload z=%.3f m, "
        "vz=%.3f m/s.",
        payload_z,
        payload_vz);

      return;
    }

    policy_start_time_ =
      now();

    first_time = false;

    RCLCPP_INFO(
      get_logger(),
      "PPO stable-hover analysis started.");
  }

  updateGlobalObservation();

  if (global_observation_.size() !=
    expected_observation_size)
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Invalid global observation size: "
      "expected %zu, received %zu.",
      expected_observation_size,
      global_observation_.size());

    return;
  }

  // Print the observation being sent to Python.
  std::ostringstream output;

  output
    << std::fixed
    << std::setprecision(9)
    << "\n"
    << "============================================================\n"
    << "         GLOBAL OBSERVATION SENT TO PYTHON\n"
    << "============================================================\n";

  // UAV observations.
  for (std::size_t uav_index = 0U;
    uav_index < num_uavs;
    ++uav_index)
  {
    const std::size_t offset =
      6U * uav_index;

    output
      << "\n"
      << uav_names_[uav_index]
      << "\n"
      << "  position_x: "
      << global_observation_[offset + 0U]
      << "\n"
      << "  position_y: "
      << global_observation_[offset + 1U]
      << "\n"
      << "  position_z: "
      << global_observation_[offset + 2U]
      << "\n"
      << "  velocity_x: "
      << global_observation_[offset + 3U]
      << "\n"
      << "  velocity_y: "
      << global_observation_[offset + 4U]
      << "\n"
      << "  velocity_z: "
      << global_observation_[offset + 5U]
      << "\n";
  }

  const std::size_t payload_offset =
    6U * num_uavs;

  const std::size_t tracking_offset =
    payload_offset + 6U;

  const std::size_t target_velocity_offset =
    tracking_offset + 3U;

  const std::size_t angle_offset =
    target_velocity_offset + 3U;

  const std::size_t additional_offset =
    angle_offset + 2U;

  output
    << "\n"
    << "Payload\n"
    << "  position_x: "
    << global_observation_[payload_offset + 0U]
    << "\n"
    << "  position_y: "
    << global_observation_[payload_offset + 1U]
    << "\n"
    << "  position_z: "
    << global_observation_[payload_offset + 2U]
    << "\n"
    << "  velocity_x: "
    << global_observation_[payload_offset + 3U]
    << "\n"
    << "  velocity_y: "
    << global_observation_[payload_offset + 4U]
    << "\n"
    << "  velocity_z: "
    << global_observation_[payload_offset + 5U]
    << "\n"
    << "\nTracking error\n"
    << "  x: "
    << global_observation_[tracking_offset + 0U]
    << "\n"
    << "  y: "
    << global_observation_[tracking_offset + 1U]
    << "\n"
    << "  z: "
    << global_observation_[tracking_offset + 2U]
    << "\n"
    << "\nTarget velocity\n"
    << "  x: "
    << global_observation_[
    target_velocity_offset + 0U]
    << "\n"
    << "  y: "
    << global_observation_[
    target_velocity_offset + 1U]
    << "\n"
    << "  z: "
    << global_observation_[
    target_velocity_offset + 2U]
    << "\n"
    << "\nPayload swing\n"
    << "  theta_x: "
    << global_observation_[angle_offset + 0U]
    << "\n"
    << "  theta_y: "
    << global_observation_[angle_offset + 1U]
    << "\n"
    << "\nCable tension comparison\n"
    << "  rope_1_distance_m: "
    << rope_tension_data_aux_[0U]
    << "\n"
    << "  rope_1_applied_tension_N: "
    << rope_tension_data_aux_[4U]
    << "\n"
    << "  rope_2_distance_m: "
    << rope_tension_data_aux_[5U]
    << "\n"
    << "  rope_2_applied_tension_N: "
    << rope_tension_data_aux_[9U]
    << "\n"
    << "  virtual_center_tension_N: "
    << virtual_total_tension_
    << "\n"
    << "  physical_resultant_tension_N: "
    << physical_total_tension_
    << "\n"
    << "\nAdditional observations\n"
    << "  total_tension_normalized: "
    << global_observation_[additional_offset + 0U]
    << "\n"
    << "  total_tension_N: "
    << global_observation_[additional_offset + 0U] *
    20.0
    << "\n"
    << "  wind_strength: "
    << global_observation_[additional_offset + 1U]
    << "\n"
    << "  policy_time: "
    << global_observation_[additional_offset + 2U]
    << "\n"
    << "============================================================\n"
    << std::defaultfloat;

  std::cout
    << output.str()
    << std::flush;

  // Verify payload odometry latency.
  const rclcpp::Time payload_stamp(
    payload_odom_aux_.header.stamp,
    get_clock()->get_clock_type());

  if (payload_stamp.nanoseconds() > 0) {
    const double latency =
      (now() - payload_stamp).seconds();

    if (latency > 1.0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Payload odometry latency is greater "
        "than 1 s: %.3f s",
        latency);
    }
  }

  // Send the observation to Python. The response
  // is processed asynchronously.
  request_ppo_shield_action();
}
//}

/* sub rope tension() //{ */
void PayloadTransport::sub_rope_tension(
  std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  constexpr std::size_t
    kExpectedRopeDataSize = 14U;

  if (msg->data.size() !=
    kExpectedRopeDataSize)
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Invalid rope tension message size: "
      "expected %zu, received %zu.",
      kExpectedRopeDataSize,
      msg->data.size());

    return;
  }

  const bool all_values_are_finite =
    std::all_of(
    msg->data.begin(),
    msg->data.end(),
    [](const double value)
    {
      return std::isfinite(value);
    });

  if (!all_values_are_finite) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Rope tension message contains a non-finite value.");

    return;
  }

  if (msg->data[13U] < 0.0) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Physical total tension cannot be negative: %.6f N.",
      msg->data[13U]);

    return;
  }

  std::lock_guard<std::mutex> lock(
    odometry_copy_mutex_);

  rope_tension_data_ =
    msg->data;

  rope_tension_received_time_ =
    std::chrono::steady_clock::now();

  first_rope_tension_received_ =
    true;
}
//}

/* sub payload odometry() //{ */
void PayloadTransport::sub_payload_odometry(
  nav_msgs::msg::Odometry::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(
    odometry_copy_mutex_);

  payload_odom_ = *msg;
  first_odometry_payload_received_ = true;
}
//}

/* sub uav odometry() //{ */
void PayloadTransport::sub_uav_odometry(
  const nav_msgs::msg::Odometry::SharedPtr msg,
  const std::size_t uav_index)
{
  std::lock_guard<std::mutex> lock(
    odometry_copy_mutex_);

  if (uav_index >= uavs_odom_.size()) {
    RCLCPP_ERROR(
      get_logger(),
      "Invalid UAV odometry index: %zu",
      uav_index);

    return;
  }

  uavs_odom_[uav_index] = *msg;
  uavs_odometry_received_[uav_index] = true;

  all_uavs_odometry_received_ =
    std::all_of(
    uavs_odometry_received_.begin(),
    uavs_odometry_received_.end(),
    [](const bool received)
    {
      return received;
    });
}
//}

}  // namespace laser_uav_multi_drones

RCLCPP_COMPONENTS_REGISTER_NODE(laser_uav_multi_drones::PayloadTransport)
