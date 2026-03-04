#!/bin/bash

set -x
set -e

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(realpath "$THIS_DIR/..")"

# ensure git and vcs are installed
# only if not already installed
if command -v git >/dev/null 2>&1; then
    echo "git already installed; skipping"
else
    sudo apt update && sudo apt install -y git
fi

if command -v vcs >/dev/null 2>&1; then
    echo "vcs already installed; skipping"
else
    sudo apt update && sudo apt install -y vcstool
fi

function clone_or_pull {
    REPO_BRANCH=$1
    REPO_URL=$2
    REPO_DIR=$3
    REPO_DIR="$PROJECT_ROOT/$REPO_DIR"
    PARENT_DIR=$(dirname "$REPO_DIR")

    cd "$PARENT_DIR"

    if [ -d "$REPO_DIR/.git" ]; then
        echo "Pulling latest changes for $REPO_DIR"
        cd "$REPO_DIR"
        git checkout "$REPO_BRANCH"
        git pull origin "$REPO_BRANCH"
        git submodule update --init --recursive
    else
        echo "Cloning $REPO_URL into $REPO_DIR"
        git clone --recursive -b "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
    fi
}

mkdir -p src

# simulator
clone_or_pull jazzy git@github.com:AustinVillaatHome/mujoco_cobot.git src/mujoco_cobot
clone_or_pull main https://github.com/ros-controls/mujoco_ros2_control.git src/mujoco_ros2_control
clone_or_pull main https://github.com/google-deepmind/mujoco_menagerie.git src/mujoco_menagerie
clone_or_pull master https://github.com/pal-robotics/mujoco_vendor.git src/mujoco_vendor
