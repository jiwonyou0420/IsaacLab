# loads franka cam env into simulator, but robot will not move, there's no control
# initially I made this to tune camera offset using GUI, but doesn't really work


import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn Franka Lift-Cube with wrist/bird cameras for tuning.")
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
from franka_cam_env_cfg import FrankaCubeLiftCamEnvCfg
# Build the env cfg and apply CLI tweaks
cfg = FrankaCubeLiftCamEnvCfg()
cfg.scene.num_envs = args_cli.num_envs

# Make the environment (no RL runner here)
env = ManagerBasedRLEnv(cfg=cfg)
env.reset()

print("\n=== FrankaCubeLiftCamEnv ===")
print("Select cameras in the Stage tree and use the gizmos in LOCAL mode:")
print("  - Wrist: /World/envs/env_0/Robot/panda_hand/wrist_cam")
print("  - Bird : /World/BirdEyeCamera")
print("Frustums are visible (debug_vis=True). Adjust OffsetCfg.pos/rot later in this file.\n")

# Keep GUI responsive
try:
    while simulation_app.is_running():
        env.sim.render()
finally:
    env.close()
    simulation_app.close()