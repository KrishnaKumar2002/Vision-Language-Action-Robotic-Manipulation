import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory('sesac_robot')
    model = LaunchConfiguration('model').perform(context)
    use_gui = LaunchConfiguration('gui').perform(context).lower() == 'true'

    xacro_files = {
        'arm': os.path.join(pkg, 'urdf', 'arm', 'arm.urdf.xacro'),
        'rover': os.path.join(pkg, 'urdf', 'rover', 'rover.urdf.xacro'),
        'rover_mesh': os.path.join(pkg, 'urdf', 'rover', 'rover_meshes.urdf.xacro'),
        'arm_mesh': os.path.join(pkg, 'urdf', 'arm', 'arm_meshes.urdf.xacro'),
    }
    rviz_files = {
        'arm': os.path.join(pkg, 'rviz', 'arm.rviz'),
        'arm_mesh': os.path.join(pkg, 'rviz', 'arm.rviz'),
        'rover': os.path.join(pkg, 'rviz', 'rover.rviz'),
        'rover_mesh': os.path.join(pkg, 'rviz', 'rover.rviz'),
    }
    if model not in xacro_files:
        raise RuntimeError(
            "model must be 'arm', 'rover', 'arm_mesh', or 'rover_mesh', not '%s'" % model)

    sesac_robot_description = xacro.process_file(
        xacro_files[model], mappings={'use_gazebo': 'false'}
    ).toxml()

    jsp_pkg = 'joint_state_publisher_gui' if use_gui else 'joint_state_publisher'
    jsp_exec = 'joint_state_publisher_gui' if use_gui else 'joint_state_publisher'

    return [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': sesac_robot_description}],
            output='screen',
        ),
        Node(package=jsp_pkg, executable=jsp_exec, output='screen'),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_files[model]],
            output='screen',
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='arm_mesh',
                              description="'arm', 'rover', 'arm_mesh', or 'rover_mesh'"),
        DeclareLaunchArgument('gui', default_value='true',
                              description='use joint_state_publisher_gui'),
        OpaqueFunction(function=launch_setup),
    ])
