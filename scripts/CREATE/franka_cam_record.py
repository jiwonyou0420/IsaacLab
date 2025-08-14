# runs inference of skrl policy in Franka Cam env
# records camera images, robot control input, etc to logs

import argparse
import os
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher
from franka_cam_inference import SKRL_MODEL_CHECKPOINT
parser = argparse.ArgumentParser(description="Play skrl checkpoint on FrankaCam env and record ALL envs.")
parser.add_argument("--checkpoint", type=str, default=SKRL_MODEL_CHECKPOINT, help="Path to skrl agent checkpoint (.pt)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--steps", type=int, default=2000, help="Max simulation steps then quit")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
simulation_app = AppLauncher(args).app

# --- imports after Kit start ---
import gymnasium as gym
import torch
import numpy as np
from PIL import Image
import csv
import skrl
from packaging import version

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.utils.io import load_yaml

from franka_cam_env_cfg import FrankaCubeLiftCamEnvCfg

if version.parse(skrl.__version__) < version.parse("1.4.2"):
    skrl.logger.error("Install skrl>=1.4.2")
    raise SystemExit(1)

if args.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
else:
    from skrl.utils.runner.jax import Runner
    skrl.config.jax.backend = "jax" if args.ml_framework == "jax" else "numpy"

def isaaclab_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def make_output_dir(num_envs: int):
    out = os.path.join(
        isaaclab_root(), "logs", "CREATE", "franka_cam_record",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(out, exist_ok=True)
    for cam in ["wrist_rgb", "wrist_depth", "bird_rgb"]:
        base = os.path.join(out, cam)
        os.makedirs(base, exist_ok=True)
        for e in range(num_envs):
            os.makedirs(os.path.join(base, f"env_{e:03d}"), exist_ok=True)
    return out

def to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    a = t.detach().cpu().numpy()
    if a.shape[-1] == 4:
        a = a[..., :3]
    return a

def depth_to_uint16_mm(t: torch.Tensor) -> np.ndarray:
    d = t.detach().cpu().numpy()
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    return np.clip(d * 1000.0, 0, 65535).astype(np.uint16)

def _load_agent_cfg_from_checkpoint(checkpoint_path: str) -> dict:
    ckpt = Path(checkpoint_path).resolve()
    yaml_path = ckpt.parent.parent / "params" / "agent.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing agent.yaml next to checkpoint: {yaml_path}")
    return load_yaml(str(yaml_path))

def main():
    import csv

    # ---- env ----
    env_cfg = FrankaCubeLiftCamEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    if args.device == "cpu":
        env_cfg.sim.use_fabric = False
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework=args.ml_framework)

    # ---- skrl runner ----
    agent_cfg = _load_agent_cfg_from_checkpoint(args.checkpoint)
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, agent_cfg)
    runner.agent.load(os.path.abspath(args.checkpoint))
    runner.agent.set_running_mode("eval")

    # ---- reset & dt ----
    obs, _ = env.reset()
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt
    N = args.num_envs

    # ---- outputs ----
    out_dir = make_output_dir(N)
    csv_path = os.path.join(out_dir, "dataset.csv")

    # CSV headers
    joint_pos_cols = [f"panda_joint{i}_pos" for i in range(1, 8)] + ["panda_finger_pos"]
    joint_vel_cols = [f"panda_joint{i}_vel" for i in range(1, 8)] + ["panda_finger_vel"]
    cmd_cols = ["goal_x", "goal_y", "goal_z"]
    action_cols = [f"action_joint{i}" for i in range(1, 8)] + ["action_finger"]

    headers = ["simulation_time", "index", "env"] + joint_pos_cols + joint_vel_cols + cmd_cols + action_cols

    # open CSV once
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(headers)

    # ---- loop ----
    for step in range(args.steps):
        if not simulation_app.is_running():
            break

        # --- get current joint pos/vel ---
        joint_pos = env.unwrapped.scene["robot"].data.joint_pos.detach().cpu().numpy()  # (N, 8)
        joint_vel = env.unwrapped.scene["robot"].data.joint_vel.detach().cpu().numpy()  # (N, 8)

        with torch.inference_mode():
            out = runner.agent.act(obs, timestep=0, timesteps=0)
            action = out[-1].get("mean_actions", out[0])  # (N, act_dim)
            obs, _, _, _, _ = env.step(action)

        # commands (goal xyz)
        try:
            cmd = env.unwrapped.command_manager.get_command("object_pose")  # (N, 7)
            goal_xyz = cmd[:, :3].detach().cpu().numpy()                    # (N, 3)
        except Exception:
            goal_xyz = np.zeros((N, 3), dtype=np.float32)

        # prepare actions
        acts = action.detach().cpu().numpy()  # (N, act_dim)
        if acts.shape[1] >= 8:
            acts8 = acts[:, :8]
        else:
            pad = np.zeros((N, 8 - acts.shape[1]), dtype=acts.dtype)
            acts8 = np.concatenate([acts, pad], axis=1)

        # write one CSV row per env
        sim_time = step * float(dt)
        for e in range(N):
            row = (
                [sim_time, step, e]
                + joint_pos[e].tolist()
                + joint_vel[e].tolist()
                + goal_xyz[e].tolist()
                + acts8[e].tolist()
            )
            writer.writerow(row)

        # images
        if args.enable_cameras:
            scene = env.unwrapped.scene
            w_out = scene["wrist_cam"].data.output
            b_out = scene["bird_cam"].data.output
            for e in range(N):
                idx_name = f"image_{step:06d}.png"

                wrist_rgb = to_uint8_rgb(w_out["rgb"][e])
                wrist_depth = depth_to_uint16_mm(w_out["distance_to_image_plane"][e])
                bird_rgb = to_uint8_rgb(b_out["rgb"][e])

                Image.fromarray(wrist_rgb).save(os.path.join(out_dir, "wrist_rgb",  f"env_{e:03d}", idx_name))
                Image.fromarray(wrist_depth).save(os.path.join(out_dir, "wrist_depth", f"env_{e:03d}", idx_name))
                Image.fromarray(bird_rgb).save(os.path.join(out_dir, "bird_rgb",   f"env_{e:03d}", idx_name))

    csv_f.close()
    print(f"[INFO] Saved CSV: {csv_path}")
    print(f"[INFO] Images in:\n  {os.path.join(out_dir, 'wrist_rgb')}\n  {os.path.join(out_dir, 'wrist_depth')}\n  {os.path.join(out_dir, 'bird_rgb')}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
