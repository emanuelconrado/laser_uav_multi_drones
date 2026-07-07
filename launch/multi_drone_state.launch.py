from launch import LaunchDescription

from launch.actions import DeclareLaunchArgument

from launch.substitutions import PathJoinSubstitution, LaunchConfiguration

from launch.actions import RegisterEventHandler, EmitEvent

from launch_ros.actions import LifecycleNode
from launch_ros.substitutions import FindPackageShare

from launch.events import matches_action
from launch.event_handlers.on_process_start import OnProcessStart
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState

import os
import lifecycle_msgs.msg

def generate_launch_description():
#Declare arguments
    declared_arguments = []
    uav_name = os.environ['UAV_NAME']
    declared_arguments.append(
        DeclareLaunchArgument(
            'multi_drone_state_file',
            default_value=PathJoinSubstitution([FindPackageShare('multi_drone_state'),
                                                'params', 'params.yaml']),
            description='Full path to the file with the all parameters.'
        )
    )

    #Initialize arguments
    manager_node_file = LaunchConfiguration('multi_drone_state_file')

    manager_node_lifecycle_node = LifecycleNode(
        package='multi_drone_state',
        executable='manager_node',
        name='manager_node',
        namespace=uav_name,
        output='screen',
        parameters=[
            manager_node_file,
            {'this_uav_name': uav_name}
        ],
        remappings=[
            ('neighbor_odom_out', '/' + uav_name + '/neighbor_velocity_position'),
            ('odometry_in', '/' + uav_name + '/ground_truth')
        ]
    )

    event_handlers = []

    event_handlers.append(
#Right after the node starts, make it take the 'configure' transition.
        RegisterEventHandler(
            OnProcessStart(
                target_action=manager_node_lifecycle_node,
                on_start=[
                    EmitEvent(event=ChangeState(
                        lifecycle_node_matcher=matches_action(manager_node_lifecycle_node),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
                    )),
                ],
            )
        ),
    )

    event_handlers.append(
        RegisterEventHandler(
            OnStateTransition(
                target_lifecycle_node=manager_node_lifecycle_node,
                start_state='configuring',
                goal_state='inactive',
                entities=[
                    EmitEvent(event=ChangeState(
                        lifecycle_node_matcher=matches_action(manager_node_lifecycle_node),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )),
                ],
            )
        ),
    )

    ld = LaunchDescription()

#Declare the arguments
    for argument in declared_arguments:
        ld.add_action(argument)

#Add client node
    ld.add_action(manager_node_lifecycle_node)

#Add event handlers
    for event_handler in event_handlers:
        ld.add_action(event_handler)

    return ld
