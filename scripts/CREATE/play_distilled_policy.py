# play_distilled_policy.py
# Run a TorchScript distilled policy (images + goal [+ optional proprio]) in Franka Cam env.

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# ---------------- CLI ----------------
parser = argparse.ArgumentParser("Play a distilled TorchScript vision policy in FrankaCam env")
parser.add_argument("--model", type=str, default="/home/andres/Documents/jiwon/IsaacLab/logs/skrl/franka_lift/2025-08-11_09-54-01_ppo_torch/checkpoints/distilled_model.pt", help="Path to distilled_model.pt (TorchScript)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--resize", type=int, default=224, help="Image size used during training")
parser.add_argument("--use_proprio", default=True, help="If student was trained with [joint_pos, joint_vel]")
parser.add_argument("--policy_device", type=str, default="cuda:0", help="Torch device for model inference")
parser.add_argument("--max_steps", type=int, default=0, help="0=run until window closes; otherwise run N steps")

# Let Isaac Lab add its usual flags, including --device (sim device)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

# Launch Kit
simulation_app = AppLauncher(args).app

# --------- Imports after Kit start ---------
import torch
import torch.nn.functional as F
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv
from franka_cam_env_cfg import FrankaCubeLiftCamEnvCfg  # your cam-enabled env


# ---------------- Helpers ----------------
def preprocess_rgb_batch(x, resize_hw, out_device):
    """
    Accepts camera output as torch Tensor [N,H,W,C] (uint8 or float in [0,1]),
    or numpy array of same shape/dtype. Returns normalized torch float
    [N,3,H',W'] on `out_device` with ImageNet mean/std.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    # x: [N,H,W,C], drop alpha if present
    if x.shape[-1] == 4:
        x = x[..., :3]

    # to float in [0,1]
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()

    # permute to [N,3,H,W]
    x = x.permute(0, 3, 1, 2).contiguous()

    # resize
    x = F.interpolate(x, size=resize_hw, mode="bilinear", align_corners=False)

    # normalize (ImageNet)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std

    return x.to(out_device, non_blocking=True)


def get_goal_xyz(env_unwrapped, N, device):
    """Use target object command (NOT true pose). Falls back to zeros."""
    try:
        cmd = env_unwrapped.command_manager.get_command("object_pose")  # (N,7)
        goal_xyz = cmd[:, :3].to(device=device)
    except Exception:
        goal_xyz = torch.zeros((N, 3), dtype=torch.float32, device=device)
    return goal_xyz


def main():
    # ---------- Build env ----------
    env_cfg = FrankaCubeLiftCamEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device  # sim device comes from AppLauncher (e.g., "cuda:0")
    if args.device == "cpu":
        env_cfg.sim.use_fabric = False

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Shortcuts
    scene = env.scene
    robot = scene["robot"]
    wrist_cam = scene["wrist_cam"]
    bird_cam  = scene["bird_cam"]

    assert wrist_cam is not None and bird_cam is not None, "Cameras not found. Ensure --enable_cameras is set."

    # ---------- Load model ----------
    policy_device = torch.device(args.policy_device if torch.cuda.is_available() else "cpu")
    ts_path = Path(args.model).resolve()
    print(f"[INFO] Loading TorchScript model: {ts_path}")
    student = torch.jit.load(str(ts_path), map_location=policy_device)
    student.eval()
    print("[INFO] Model loaded.")

    # ---------- Reset ----------
    obs, _ = env.reset()
    N = args.num_envs

    # Use env's tensor device for actions to avoid copies
    sim_tensor_device = robot.data.joint_pos.device

    # Step loop
    step = 0
    print("[INFO] Starting control loop...")
    while simulation_app.is_running():
        if args.max_steps > 0 and step >= args.max_steps:
            break

        # Camera tensors are already torch [N,H,W,C]
        w_rgb = wrist_cam.data.output["rgb"]
        b_rgb = bird_cam.data.output["rgb"]

        wrist = preprocess_rgb_batch(w_rgb, (args.resize, args.resize), policy_device)
        bird  = preprocess_rgb_batch(b_rgb,  (args.resize, args.resize), policy_device)

        # Target object position (command), not privileged real pose
        goal_xyz = get_goal_xyz(env.unwrapped, N, policy_device)

        # Optional proprio
        proprio = None
        if args.use_proprio:
            jpos = robot.data.joint_pos  # (N,J)
            jvel = robot.data.joint_vel  # (N,J)
            proprio = torch.cat([jpos, jvel], dim=1).to(policy_device, non_blocking=True)

        # Inference
        with torch.inference_mode():
            if proprio is None:
                actions = student(wrist, bird, goal_xyz)          # [N,A]
            else:
                actions = student(wrist, bird, goal_xyz, proprio) # [N,A]

        # Ensure correct dtype/device for env
        actions = actions.to(sim_tensor_device, dtype=torch.float32, non_blocking=True)

        # Step env
        obs, rew, term, trunc, info = env.step(actions)
        step += 1

    print("[INFO] Closing env...")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
