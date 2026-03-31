from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params = PathJoinSubstitution(
        [FindPackageShare("rosbot_deepc"), "config", "deepc_params.yaml"]
    )

    return LaunchDescription([
        Node(
            package="rosbot_deepc",
            executable="deepc_node",
            name="deepc_node",
            output="screen",
            parameters=[params]
        ),
    ])