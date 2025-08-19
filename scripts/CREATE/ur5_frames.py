# viz_frames_ur5.py
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("UR5 + XArm gripper frame viz")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

simulation_app = AppLauncher(args).app

# ---- imports AFTER Kit starts ----
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG

# your lift env cfg with UR5 + XArm
from isaaclab_tasks.manager_based.manipulation.lift.config.ur_5.joint_pos_env_cfg import UR5CubeLiftEnvCfg

# ---------- build a clean InteractiveSceneCfg from your env's scene ----------
env_cfg = UR5CubeLiftEnvCfg()
scene_cfg = env_cfg.scene.copy()          # IMPORTANT: copy, don't mutate the RL env config in-place
scene_cfg.num_envs = args.num_envs

# (Optional) ensure a ground plane/light exist in the copied scene if your env omitted them.
# scene_cfg.ground = scene_cfg.ground or AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
# scene_cfg.dome_light = scene_cfg.dome_light or AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000))

# ---------- add frame transformers (names are arbitrary keys on the scene cfg) ----------
marker = FRAME_MARKER_CFG.copy()
marker.prim_path = "/Visuals/FrameTransformer"
marker.markers["frame"].scale = (0.08, 0.08, 0.08)

# Base ↔ EE (use your actual link names as seen in the stage)
scene_cfg.ee_frames = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/ur5/base_link",
    target_frames=[
        FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/ur5/wrist_3_link", name="wrist3"),
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/xarm_gripper/xarm_gripper_base_link",
            name="gripper_base",
        ),
        # quick TCP visualization ~7 cm along the gripper forward axis (tune as needed)
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/xarm_gripper/xarm_gripper_base_link",
            name="tcp",
        ),
    ],
    debug_vis=True,
    visualizer_cfg=marker,
)

# Base ↔ Cube
# scene_cfg.cube_frames = FrameTransformerCfg(
#     prim_path="{ENV_REGEX_NS}/Robot/ur5/base_link",
#     target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Object", name="cube")],
#     debug_vis=True,
#     visualizer_cfg=marker,
# )

# ---------- run a tiny stepping loop ----------
def run(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        if count % 240 == 0:
            # reset root pose to env origins
            root = scene["robot"].data.default_root_state.clone()
            root[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root[:, :7])
            scene["robot"].write_root_velocity_to_sim(root[:, 7:])
            # set slightly noisy joint positions
            q = scene["robot"].data.default_joint_pos.clone()
            dq = scene["robot"].data.default_joint_vel.clone()
            q += torch.rand_like(q) * 0.05
            scene["robot"].write_joint_state_to_sim(q, dq)
            scene.reset()
            print("[viz] reset")

        targets = scene["robot"].data.default_joint_pos
        scene["robot"].set_joint_position_target(targets)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1

        # quick sanity prints (comment out if chatty)
        # print("EE->targets pos (base frame):", scene["ee_frames"].data.target_pos_source)

def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO] Frame viz scene ready.")
    run(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
