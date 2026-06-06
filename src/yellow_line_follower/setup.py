from setuptools import setup

package_name = 'yellow_line_follower'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='esy',
    maintainer_email='esy@todo.todo',
    description='Yellow line follower for TurtleBot3 single yellow line track',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yellow_line_follower_node = yellow_line_follower.yellow_line_follower_node:main',
        ],
    },
)
