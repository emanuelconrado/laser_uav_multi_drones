import os

import lifecycle_msgs.msg
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    declared_arguments = []

    uav_name = os.environ["UAV_NAME"]

    declared_arguments.append(
        DeclareLaunchArgument(
            "payload_transport_file",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("laser_uav_multi_drones"),
                    "params",
                    "payload_transport.yaml",
                ]
            ),
            description="Full path to the payload transport parameter file.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value=PythonExpression(
                ['"', os.getenv("REAL_UAV", "true"), '" == "false"']
            ),
            description="Whether to use simulation time.",
        )
    )

    # Initialize arguments
    payload_transport_file = LaunchConfiguration("payload_transport_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # Payload transport lifecycle node
    payload_transport_node = LifecycleNode(
        package="laser_uav_multi_drones",
        executable="payload_transport_node",
        name="payload_transport",
        namespace=uav_name,
        output="screen",
        parameters=[
            payload_transport_file,
            {"this_uav_name": uav_name},
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            (
                "neighbor_odom_out",
                "/" + uav_name + "/neighbor_velocity_position",
            ),
            (
                "odometry_in",
                "/" + uav_name + "/ground_truth",
            ),
            (
                "odometry_payload_in",
                "/payload/ground_truth",
            ),
            (
                "action_out",
                "/" + uav_name + "/action",
            ),
        ],
    )

    # Lifecycle event handlers
    event_handlers = []

    # Configure node after process starts
    event_handlers.append(
        RegisterEventHandler(
            OnProcessStart(
                target_action=payload_transport_node,
                on_start=[
                    EmitEvent(
                        event=ChangeState(
                            lifecycle_node_matcher=matches_action(
                                payload_transport_node
                            ),
                            transition_id=(
                                lifecycle_msgs.msg.Transition.
                                TRANSITION_CONFIGURE
                            ),
                        )
                    ),
                ],
            )
        )
    )

    # Activate node after successful configuration
    event_handlers.append(
        RegisterEventHandler(
            OnStateTransition(
                target_lifecycle_node=payload_transport_node,
                start_state="configuring",
                goal_state="inactive",
                entities=[
                    EmitEvent(
                        event=ChangeState(
                            lifecycle_node_matcher=matches_action(
                                payload_transport_node
                            ),
                            transition_id=(
                                lifecycle_msgs.msg.Transition.
                                TRANSITION_ACTIVATE
                            ),
                        )
                    ),
                ],
            )
        )
    )

    ld = LaunchDescription()

    # Declare arguments
    for argument in declared_arguments:
        ld.add_action(argument)

    # Add lifecycle node
    ld.add_action(payload_transport_node)

    # Add lifecycle event handlers
    for event_handler in event_handlers:
        ld.add_action(event_handler)

    return ld
