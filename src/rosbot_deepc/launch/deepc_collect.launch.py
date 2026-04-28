from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def config_file(name: str) -> PathJoinSubstitution:
    return PathJoinSubstitution([
        FindPackageShare('rosbot_deepc'),
        'config',
        name,
    ])


def generate_launch_description():
    mode = LaunchConfiguration('mode')
    common_params = [
        config_file('common_runtime.yaml'),
        config_file('common_collect.yaml'),
    ]

    reference_node = Node(
        package='rosbot_deepc',
        executable='reference_collect_node',
        name='reference_collect_node',
        output='screen',
        parameters=[
            *common_params,
            config_file('reference_collect.yaml'),
        ],
        condition=IfCondition(PythonExpression(["'", mode, "' == 'reference'"]))
    )

    random_node = Node(
        package='rosbot_deepc',
        executable='random_collect_node',
        name='random_collect_node',
        output='screen',
        parameters=[
            *common_params,
            config_file('random_collect.yaml'),
        ],
        condition=IfCondition(PythonExpression(["'", mode, "' == 'random'"]))
    )

    prbs_node = Node(
        package='rosbot_deepc',
        executable='prbs_collect_node',
        name='prbs_collect_node',
        output='screen',
        parameters=[
            *common_params,
            config_file('prbs_collect.yaml'),
        ],
        condition=IfCondition(PythonExpression(["'", mode, "' == 'prbs'"]))
    )

    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='reference'),
        reference_node,
        random_node,
        prbs_node,
    ])
