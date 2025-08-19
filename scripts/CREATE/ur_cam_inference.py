# loads lift cube policy(skrl, RL trained) into FrankaCam env
# keeps running inference
# easier to tune camera offsets from GUI
# CLI arguments such as checkpoint


import argparse
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# ---------------- CLI ----------------
SKRL_MODEL_CHECKPOINT = "/home/andres/Documents/jiwon/IsaacLab/logs/skrl/ur10_lift/2025-08-15_15-59-45_ppo_torch/checkpoints/best_agent.pt"
parser = argparse.ArgumentParser(description="Play a skrl checkpoint on a custom URLiftCam env (no registry).")
parser.add_argument("--checkpoint", type=str, default=SKRL_MODEL_CHECKPOINT, help="Path to skrl agent checkpoint (.pt)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument(
    "--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"], help="skrl backend"
)
# Isaac Lab / Kit args (adds --device, --renderer, --headless, --enable_cameras, etc.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # required to actually spawn camera sensors
# Cameras optional; set True if you want sensors in your env cfg to spawn
# args.enable_cameras = True

# Launch Kit
simulation_app = AppLauncher(args).app

# --------- Imports after Kit start ---------
import gymnasium as gym
import torch
import skrl
from packaging import version

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.utils.io import load_yaml

# >>>>>> CHANGE THIS to your module / class name <<<<<<
from ur_cam_env_cfg import UR10CubeLiftCamEnvCfg# <-- your custom env cfg

if version.parse(skrl.__version__) < version.parse("1.4.2"):
    skrl.logger.error("Unsupported skrl version. Please install skrl>=1.4.2")
    raise SystemExit(1)

if args.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
else:
    from skrl.utils.runner.jax import Runner
    skrl.config.jax.backend = "jax" if args.ml_framework == "jax" else "numpy"


def _load_agent_cfg_from_checkpoint(checkpoint_path: str) -> dict:
    """Given .../checkpoints/best_agent.pt, load the agent config from .../params/agent.yaml"""
    ckpt = Path(checkpoint_path).resolve()
    # logs/.../<run_dir>/checkpoints/best_agent.pt -> logs/.../<run_dir>/params/agent.yaml
    params_dir = ckpt.parent.parent / "params"
    yaml_path = params_dir / "agent.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"agent.yaml not found next to checkpoint.\nExpected at: {yaml_path}\n"
            "Make sure you trained with Isaac Lab’s skrl runner (it dumps params/agent.yaml)."
        )
    return load_yaml(str(yaml_path))


def main():
    # 1) Build env cfg directly (no registry), then create env
    env_cfg = UR10CubeLiftCamEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    if args.device == "cpu":
        env_cfg.sim.use_fabric = False  # USD I/O path on CPU
    # from franka_light_dr import apply_light_randomization
    # apply_light_randomization(
    #     env_cfg
    # )
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # (optional) wrap for gym video here if you want (not included for brevity)

    # 2) Wrap for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args.ml_framework)

    # 3) Build a skrl Runner using the agent config loaded from the checkpoint’s params folder
    agent_cfg = _load_agent_cfg_from_checkpoint(args.checkpoint)
    # Don’t log or checkpoint anything while playing
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    runner = Runner(env, agent_cfg)

    # 4) Load checkpoint and set eval
    ckpt = os.path.abspath(args.checkpoint)
    print(f"[INFO] Loading checkpoint: {ckpt}")
    runner.agent.load(ckpt)
    runner.agent.set_running_mode("eval")

    # 5) Step loop
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    obs, _ = env.reset()

    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            out = runner.agent.act(obs, timestep=0, timesteps=0)
            # Single-agent deterministic action: prefer mean_actions if present
            action = out[-1].get("mean_actions", out[0])
            obs, _, _, _, _ = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
