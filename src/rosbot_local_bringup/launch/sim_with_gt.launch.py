from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    gz_world = LaunchConfiguration('gz_world')
    world_name = LaunchConfiguration('world_name')
    robot_model = LaunchConfiguration('robot_model')
    rviz = LaunchConfiguration('rviz')
    gz_headless_mode = LaunchConfiguration('gz_headless_mode')
    reset_z = LaunchConfiguration('reset_z')

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

    set_pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='set_pose_bridge',
        output='screen',
        arguments=[[
            TextSubstitution(text='/world/'),
            world_name,
            TextSubstitution(text='/set_pose@ros_gz_interfaces/srv/SetEntityPose'),
        ]],
        parameters=[{'use_sim_time': True}],
    )

    control_world_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='control_world_bridge',
        output='screen',
        arguments=[[
            TextSubstitution(text='/world/'),
            world_name,
            TextSubstitution(text='/control@ros_gz_interfaces/srv/ControlWorld'),
        ]],
        parameters=[{'use_sim_time': True}],
    )

    ready_notifier = Node(
        package='rosbot_local_bringup',
        executable='sim_ready_notifier',
        name='sim_ready_notifier',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    reset_server = Node(
        package='rosbot_local_bringup',
        executable='reset_rosbot_server',
        name='reset_rosbot_server',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'reset_service': '/reset_rosbot',
            'cmd_vel_topic': '/cmd_vel',
            'set_pose_service': [
                TextSubstitution(text='/world/'),
                world_name,
                TextSubstitution(text='/set_pose'),
            ],
            'entity_name': 'rosbot',
            'frame_id': 'base_link',
            'target_z': ParameterValue(reset_z, value_type=float),
            'pre_zero_publish_count': 3,
            'post_zero_publish_count': 5,
            'zero_publish_period': 0.03,
            'service_timeout_sec': 1.0,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'gz_world',
            default_value='/ws/src/worlds/flat_empty.sdf'
        ),
        DeclareLaunchArgument(
            'world_name',
            default_value='flat_empty'
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
        DeclareLaunchArgument(
            'reset_z',
            default_value='0.1'
        ),
        sim,
        gt_bridge,
        set_pose_bridge,
        control_world_bridge,
        ready_notifier,
        reset_server,
    ])
