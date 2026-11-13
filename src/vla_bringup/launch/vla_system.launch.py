import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Use the realistic robot arm from sesac_robot
    sesac_robot_pkg = get_package_share_directory('sesac_robot')
    xacro_file = os.path.join(sesac_robot_pkg, 'urdf', 'arm', 'arm_meshes.urdf.xacro')
    robot_desc = xacro.process_file(xacro_file, mappings={'use_gazebo': 'false'}).toxml()
    
    vla_desc_pkg = get_package_share_directory('vla_description')
    rviz_file = os.path.join(vla_desc_pkg, 'rviz', 'vla_scene.rviz')

    return LaunchDescription([
        # 1. Robot State Publisher for sesac_robot
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        ),
        # 2. Pre-configured RViz2 Window
        Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', rviz_file]
        ),
        # 3. Overhead RGB-D Perception Camera Stream Node (Zero hardcoded coordinates)
        Node(
            package='vla_perception',
            executable='camera_stream_node',
            output='screen'
        ),
        # 4. Octo-VLA End-to-End Neural Policy Node
        Node(
            package='vla_perception',
            executable='vla_policy_node',
            output='screen'
        ),
        # 5. Dynamic Camera-Driven Object Detector (Backward compatibility & Grounding)
        Node(
            package='vla_perception',
            executable='detector_node',
            output='screen'
        ),
        # 6. Language Task Planner
        Node(
            package='vla_task_planner',
            executable='language_task_node',
            output='screen'
        ),
        # 7. Collision-Aware Trajectory & Closed-Loop Action Executor
        Node(
            package='vla_motion_planning',
            executable='trajectory_executor_node',
            output='screen'
        ),
    ])
