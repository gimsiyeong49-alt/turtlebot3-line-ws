import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('tb3_cam10_bringup')

    camera_yaml = os.path.join(share, 'config', 'camera_320x240_10hz.yaml')
    camset_script = os.path.join(share, 'scripts', 'camset.sh')

    cam_dev = '/dev/video4'

    return LaunchDescription([
        LogInfo(msg=f'[tb3_cam10_bringup] camera device: {cam_dev}'),

        ExecuteProcess(
            cmd=['bash', '-lc', 'pkill -f "[u]sb_cam" || true; pkill -f "[v]4l2_camera" || true'],
            output='screen'
        ),

        ExecuteProcess(
            cmd=['v4l2-ctl', '-d', cam_dev, '--set-parm=10'],
            output='screen'
        ),

        ExecuteProcess(
            cmd=['bash', camset_script],
            output='screen'
        ),

        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package='usb_cam',
                    executable='usb_cam_node_exe',
                    name='usb_cam',
                    parameters=[
                        camera_yaml,
                        {'video_device': cam_dev},
                    ],
                    output='screen',
                    emulate_tty=True,
                )
            ]
        ),

        TimerAction(
            period=2.5,
            actions=[
                ExecuteProcess(
                    cmd=['bash', camset_script],
                    output='screen'
                )
            ]
        ),
    ])
