# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Lift-Cube-UR5-Adapt-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR5AdaptHandCubeLiftEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",

    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-UR5-Adapt-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR5AdaptHandCubeLiftEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)


