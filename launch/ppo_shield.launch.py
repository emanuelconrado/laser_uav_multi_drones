#!/usr/bin/env python3

import sys

from pathlib import Path

from ament_index_python.packages import (
    get_package_prefix,
)
from launch import LaunchDescription
from launch.actions import ExecuteProcess


PACKAGE_NAME = "laser_uav_multi_drones"


def generate_launch_description():
    package_prefix = Path(
        get_package_prefix(
            PACKAGE_NAME
        )
    )

    python_executable = Path(
        sys.executable
    )

    installed_node = (
        package_prefix
        / "lib"
        / PACKAGE_NAME
        / "ppo_shield_node.py"
    )

    if not python_executable.is_file():
        raise RuntimeError(
            "Python executable was not found: "
            f"{python_executable}"
        )

    if not installed_node.is_file():
        raise RuntimeError(
            "Installed PPO shield node was "
            "not found: "
            f"{installed_node}"
        )

    ppo_shield_process = ExecuteProcess(
        name="ppo_shield_node",
        cmd=[
            str(python_executable),
            str(installed_node),
        ],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        ppo_shield_process,
    ])
