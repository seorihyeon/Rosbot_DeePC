from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('rosbot_local_bringup'),
                'launch',
                'sim_with_gt.launch.py'
            ])
        )
    )

    collector = Node(
        package='rosbot_deepc',
        executable='collect_node',
        name='collect_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('rosbot_deepc'),
                'config',
                'collect.yaml'
            ]),
            {'use_sim_time': True}
        ]
    )

    return LaunchDescription([
        sim,
        collector,
    ])