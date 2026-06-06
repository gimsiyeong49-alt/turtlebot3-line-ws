import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('tb3_line_bringup')

    line_yaml = os.path.join(share, 'config', 'line_follower.yaml')
    lidar_yaml = os.path.join(share, 'config', 'lidar_mux_yaw.yaml')

    return LaunchDescription([
        Node(
            package='yellow_line_follower',
            executable='yellow_line_follower_node',
            name='yellow_line_follower_node',
            parameters=[line_yaml],
            output='screen',
            emulate_tty=True,
        ),

        TimerAction(
            period=0.5,
            actions=[
                Node(
                    package='tb3_line_bringup',
                    executable='angle_lidar_mux_yaw',
                    name='angle_lidar_mux',
                    parameters=[lidar_yaml],
                    output='screen',
                    emulate_tty=True,
                )
            ]
        ),
    ])
