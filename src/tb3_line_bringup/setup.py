import os
from glob import glob
from setuptools import setup

package_name = 'tb3_line_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='turtlebot3_7',
    maintainer_email='turtlebot3_7@todo.todo',
    description='TurtleBot3 line follower bringup and lidar mux',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'angle_lidar_mux_yaw = tb3_line_bringup.angle_lidar_mux_yaw:main',
        ],
    },
)
