# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold

# a custom termination condition for the object being at the goal and stable
# for early termination during inference

def object_at_goal_stable(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    pos_threshold: float = 0.02,          # meters
    lin_vel_threshold: float = 0.05,      # m/s
    ang_vel_threshold: float = 0.50,      # rad/s
    hold_time_s: float = 0.5,             # must remain stable this long
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Terminate when the object is within `pos_threshold` of the goal AND
    its linear & angular velocities are small, maintained for `hold_time_s`."""
    device = env.device
    num_envs = env.num_envs

    # lazy-init per-env counters (survive across steps, reset by env.reset())
    if not hasattr(env, "_stable_goal_counter"):
        env._stable_goal_counter = torch.zeros(num_envs, dtype=torch.int32, device=device)

    # how many steps we must hold
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt
    hold_steps = max(1, int(round(hold_time_s / dt)))

    # scene handles
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    # goal in world
    goal_pos_b = cmd[:, :3]
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b)

    # current object state
    obj_pos_w = obj.data.root_pos_w[:, :3]
    obj_lin_vel = obj.data.root_lin_vel_w
    obj_ang_vel = obj.data.root_ang_vel_w

    # instantaneous “in goal & stable”
    close = torch.norm(goal_pos_w - obj_pos_w, dim=1) < pos_threshold
    # angular velocity has errors, just use linear velocity
    slow = (obj_lin_vel.norm(dim=1) < lin_vel_threshold)
    ok_now = close & slow

    # update counters: +1 if ok, else 0
    env._stable_goal_counter = torch.where(ok_now, env._stable_goal_counter + 1, torch.zeros_like(env._stable_goal_counter))

    # stable goal counter debugging
    # print("dist")
    # print(torch.norm(goal_pos_w - obj_pos_w, dim=1))
    # print("pos_threshold")
    # print(pos_threshold)
    # print("obj_lin_vel")
    # print(obj_lin_vel.norm(dim=1))
    # print("obj_lin_vel threshold")
    # print(lin_vel_threshold)
    # print("ok_now")
    # print(ok_now)
    # print("env._stable_goal_counter")
    # print(env._stable_goal_counter)
    return env._stable_goal_counter >= hold_steps