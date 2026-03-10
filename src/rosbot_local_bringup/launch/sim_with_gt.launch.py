from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    gz_world = LaunchConfiguration('gz_world')
    robot_model = LaunchConfiguration('robot_model')
    rviz = LaunchConfiguration('rviz')
    gz_headless_mode = LaunchConfiguration('gz_headless_mode')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('rosbot_gazebo'),
                'launch',
                'simulation.launch.py'
            ])
        ),
        launch_arguments={
            'use_sim': 'True',
            'robot_model': robot_model,
            'rviz': rviz,
            'gz_headless_mode': gz_headless_mode,
            'gz_world': gz_world,
        }.items()
    )

    gt_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ground_truth_bridge',
        output='screen',
        arguments=[
            '/model/rosbot/odometry_gt@nav_msgs/msg/Odometry[gz.msgs.Odometry'
        ],
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'gz_world',
            default_value='/ws/src/worlds/flat_empty.sdf'
        ),
        DeclareLaunchArgument(
            'robot_model',
            default_value='rosbot'
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='False'
        ),
        DeclareLaunchArgument(
            'gz_headless_mode',
            default_value='True'
        ),
        sim,
        gt_bridge,
    ])