cmake_minimum_required(VERSION 3.8)
project(laser_uav_multi_drones)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(laser_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

set(dependencies
    Eigen3
    rclcpp
    rclcpp_lifecycle
    rclcpp_components
    nav_msgs
    laser_msgs
    tf2_geometry_msgs
    )

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  )

# Drone avoidance node
set(drone_avoidance_executable_name drone_avoidance_node)
set(drone_avoidance_library_name ${PROJECT_NAME}_drone_avoidance_core)

add_executable(
  ${drone_avoidance_executable_name}
  src/drone_avoidance_main.cpp
  )

add_library(
  ${drone_avoidance_library_name}
  SHARED
  src/drone_avoidance_node.cpp
  )

ament_target_dependencies(
  ${drone_avoidance_executable_name}
  ${dependencies}
  )

ament_target_dependencies(
  ${drone_avoidance_library_name}
  ${dependencies}
  )

target_link_libraries(
  ${drone_avoidance_executable_name}
  ${drone_avoidance_library_name}
  )

target_include_directories(
  ${drone_avoidance_library_name}
  PRIVATE
  include
  )

rclcpp_components_register_nodes(
  ${drone_avoidance_library_name}
  "drone_avoidance_node_cpp::DroneAvoidance"
  )

# Payload transport node
set(payload_transport_executable_name payload_transport_node)
set(payload_transport_library_name ${PROJECT_NAME}_payload_transport_core)

add_executable(
  ${payload_transport_executable_name}
  src/payload_transport_main.cpp
  )

add_library(
  ${payload_transport_library_name}
  SHARED
  src/payload_transport_node.cpp
  )

ament_target_dependencies(
  ${payload_transport_executable_name}
  ${dependencies}
  )

ament_target_dependencies(
  ${payload_transport_library_name}
  ${dependencies}
  )

target_link_libraries(
  ${payload_transport_executable_name}
  ${payload_transport_library_name}
  )

target_include_directories(
  ${payload_transport_library_name}
  PRIVATE
  include
  )

rclcpp_components_register_nodes(
  ${payload_transport_library_name}
  "payload_transport_node_cpp::PayloadTransport"
  )

# Install libraries
install(
  TARGETS
    ${drone_avoidance_library_name}
    ${payload_transport_library_name}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )

# Install executables
install(
  TARGETS
    ${drone_avoidance_executable_name}
    ${payload_transport_executable_name}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  )

install(
  DIRECTORY include/
  DESTINATION include/
  )

install(
  DIRECTORY launch params
  DESTINATION share/${PROJECT_NAME}
  )

if (BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif ()

ament_export_include_directories(include)
ament_export_dependencies(${dependencies})

ament_package()
