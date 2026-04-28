from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def config_file(name: str) -> PathJoinSubstitution:
    return PathJoinSubstitution(
        [FindPackageShare("rosbot_deepc"), "config", name]
    )


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="rosbot_deepc",
            executable="deepc_node",
            name="deepc_node",
            output="screen",
            parameters=[
                config_file("common_runtime.yaml"),
                config_file("common_tracking.yaml"),
                config_file("deepc_params.yaml"),
            ],
        ),
    ])
