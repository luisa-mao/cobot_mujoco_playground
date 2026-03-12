#!/bin/bash

set -e  # Exit on any error
set -x  # Print commands as they execute

ROS_DISTRO=jazzy

# =============================================================================
# ROS2 Package Dependencies (MuJoCo Simulator + MoveIt + ros2_control)
# =============================================================================
apt-get update
apt-get install -y \
  ros-dev-tools \
  ros-$ROS_DISTRO-foxglove-bridge \
  ros-$ROS_DISTRO-rviz2 \
  ros-$ROS_DISTRO-joint-state-publisher \
  ros-$ROS_DISTRO-joint-state-publisher-gui \
  ros-$ROS_DISTRO-rmw-cyclonedds-cpp \
  ros-$ROS_DISTRO-test-msgs \
  ros-$ROS_DISTRO-backward-ros \
  ros-$ROS_DISTRO-generate-parameter-library \
  ros-$ROS_DISTRO-generate-parameter-library-py \
  ros-$ROS_DISTRO-control-toolbox \
  ros-$ROS_DISTRO-kinematics-interface \
  ros-$ROS_DISTRO-kinematics-interface-kdl \
  ros-$ROS_DISTRO-orocos-kdl-vendor \
  ros-$ROS_DISTRO-kdl-parser \
  ros-$ROS_DISTRO-eigen-stl-containers \
  ros-$ROS_DISTRO-eigen3-cmake-module \
  ros-$ROS_DISTRO-geometric-shapes \
  ros-$ROS_DISTRO-random-numbers \
  ros-$ROS_DISTRO-shape-msgs \
  ros-$ROS_DISTRO-osqp-vendor \
  ros-$ROS_DISTRO-ruckig \
  ros-$ROS_DISTRO-srdfdom \
  ros-$ROS_DISTRO-ament-cmake-google-benchmark \
  ros-$ROS_DISTRO-ros-testing \
  ros-$ROS_DISTRO-launch-param-builder \
  ros-$ROS_DISTRO-topic-tools \
  ros-$ROS_DISTRO-object-recognition-msgs \
  ros-$ROS_DISTRO-launch-pytest \
  ros-$ROS_DISTRO-warehouse-ros-sqlite \
  ros-$ROS_DISTRO-warehouse-ros \
  ros-$ROS_DISTRO-ament-clang-format \
  ros-$ROS_DISTRO-ament-clang-tidy \
  ros-$ROS_DISTRO-stomp \
  ros-$ROS_DISTRO-ompl \
  ros-$ROS_DISTRO-rmf-utils \
  ros-$ROS_DISTRO-tf-transformations \
  ros-$ROS_DISTRO-realtime-tools \
  ros-$ROS_DISTRO-pal-statistics \
  ros-$ROS_DISTRO-gripper-controllers \
  ros-$ROS_DISTRO-cv-bridge \
  ros-$ROS_DISTRO-vision-opencv \
  ros-$ROS_DISTRO-moveit-msgs \
  ros-$ROS_DISTRO-moveit-resources \
  ros-$ROS_DISTRO-moveit \
  ros-$ROS_DISTRO-moveit-servo \
  ros-$ROS_DISTRO-ros2-control \
  ros-$ROS_DISTRO-ros2-control-cmake \
  ros-$ROS_DISTRO-robotiq-description \
  ros-$ROS_DISTRO-ros2-controllers \
  ros-$ROS_DISTRO-ament-index-cpp \
  ros-$ROS_DISTRO-nav-msgs
