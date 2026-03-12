from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    pkg_share = FindPackageShare(package='guge_urdf').find('guge_urdf')
    urdf_file = os.path.join(pkg_share, 'urdf', 'guge_urdf.urdf')
    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare("gazebo_ros"), '/launch', '/gazebo.launch.py'
        ]),
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'guge_urdf', '-file', urdf_file],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        spawn_entity,
    ])
