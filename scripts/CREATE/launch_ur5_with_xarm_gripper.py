# for checking default ur5 with xarm gripper env


import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn ur5 with xarm gripper for scene inspection.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
# App/Kit args (adds --device, --renderer, --headless, --enable_cameras, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # required to actually spawn camera sensors

# Launch Kit
simulation_app = AppLauncher(args_cli).app

# Imports that require Kit to be up
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift.config.ur_5.joint_pos_env_cfg import UR5CubeLiftEnvCfg
# Build the env cfg and apply CLI tweaks
cfg = UR5CubeLiftEnvCfg()
cfg.scene.num_envs = args_cli.num_envs

# Make the environment (no RL runner here)
env = ManagerBasedRLEnv(cfg=cfg)
env.reset()


# Keep GUI responsive
try:
    while simulation_app.is_running():
        env.sim.render()
finally:
    env.close()
    simulation_app.close()