cmake_minimum_required(VERSION 3.8)
project(sesac_robot)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf launch rviz meshes config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
