# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

# markers for optional EE frame viz
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.ur5_adapt_hand import UR5_ADAPT_HAND_CFG  # isort: skip


# ---- Names we’ll use repeatedly ----
UR5_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Only NON‑MIMIC hand joints here (masters you actually want to command).
# If some of these are mimics in your URDF, simply remove them.
ADAPT_HAND_JOINTS = [
    # thumb
    "Thumb_CMC1", "Thumb_CMC2", "Thumb_MCP", "Thumb_IP",
    # index
    "Index_MCP", "Index_PIP", "Index_DIP",
    # middle
    "Middle_MCP", "Middle_PIP", "Middle_DIP",
    # ring
    "Ring_MCP", "Ring_PIP", "Ring_DIP",
    # pinky
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP",
    # spreads (set to 0.0 in both open/close unless you want splay)
    "Index_MCP_Spread", "Middle_MCP_Spread", "Ring_MCP_Spread", "Pinky_MCP_Spread",
    # wrist joints of the hand (leave neutral in binary open/close for now)
    "Wrist_Pitch", "Wrist_Yaw",
]

# Binary postures: start conservative; tune after visual check.
# Open = small extension and no spread; Close = curl fingers in.
OPEN_POSE = {
    # thumb (slightly extended/abducted)
    "Thumb_CMC1":  0.04, "Thumb_CMC2": -0.04,
    "Thumb_MCP":  -0.04, "Thumb_IP":   -0.04,
    # fingers (near straight)
    "Index_MCP":  -0.02, "Index_PIP":  -0.02, "Index_DIP":  -0.02,
    "Middle_MCP": -0.02, "Middle_PIP": -0.02, "Middle_DIP": -0.02,
    "Ring_MCP":   -0.02, "Ring_PIP":   -0.02, "Ring_DIP":   -0.02,
    "Pinky_MCP":  -0.02, "Pinky_PIP":  -0.02, "Pinky_DIP":  -0.02,
    # spreads (neutral)
    "Index_MCP_Spread": 0.0, "Middle_MCP_Spread": 0.0,
    "Ring_MCP_Spread":  0.0, "Pinky_MCP_Spread":  0.0,
    # hand wrist neutral
    "Wrist_Pitch": 0.0, "Wrist_Yaw": 0.0,
}

CLOSE_POSE = {
    # thumb (flex & oppose)
    "Thumb_CMC1":  0.25, "Thumb_CMC2":  0.15,
    "Thumb_MCP":   0.35, "Thumb_IP":    0.35,
    # fingers (curl)
    "Index_MCP":   0.60, "Index_PIP":   0.80, "Index_DIP":   0.70,
    "Middle_MCP":  0.60, "Middle_PIP":  0.80, "Middle_DIP":  0.70,
    "Ring_MCP":    0.55, "Ring_PIP":    0.75, "Ring_DIP":    0.65,
    "Pinky_MCP":   0.50, "Pinky_PIP":   0.70, "Pinky_DIP":   0.60,
    # spreads (keep zero for a firm pinch; set >0.0 if you want splay)
    "Index_MCP_Spread": 0.0, "Middle_MCP_Spread": 0.0,
    "Ring_MCP_Spread":  0.0, "Pinky_MCP_Spread":  0.0,
    # hand wrist neutral for binary; later you can exclude these joints entirely
    "Wrist_Pitch": 0.0, "Wrist_Yaw": 0.0,
}


@configclass
class UR5AdaptHandCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---------- Robot ----------
        self.scene.robot = UR5_ADAPT_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ---------- Actions ----------
        # 6‑DoF UR5 arm in joint space
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=UR5_ARM_JOINTS,
            scale=0.5,
            use_default_offset=True,
        )


        # Continuous Adapt hand (drive non‑mimic DOFs only)
        # HAND_DRIVE_JOINTS = [
        #     # thumb
        #     "Thumb_CMC1", "Thumb_CMC2", "Thumb_MCP", "Thumb_IP",
        #     # fingers (drive PIP + MCP; drop DIP if they’re mimic in your URDF)
        #     "Index_MCP", "Index_PIP",
        #     "Middle_MCP", "Middle_PIP",
        #     "Ring_MCP", "Ring_PIP",
        #     "Pinky_MCP", "Pinky_PIP",
        #     # spreads (optional)
        #     "Index_MCP_Spread", "Middle_MCP_Spread", "Ring_MCP_Spread", "Pinky_MCP_Spread",
        #     # hand wrist (optional)
        #     "Wrist_Pitch", "Wrist_Yaw",
        # ]

        # self.actions.gripper_action = mdp.JointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=HAND_DRIVE_JOINTS,
        #     scale=0.5,                 # tune
        #     use_default_offset=True,
        # )

        # Binary "gripper" mapped to Adapt hand: open vs close postures
        # NOTE: LiftEnv expects BinaryJointPositionActionCfg here.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=ADAPT_HAND_JOINTS,
            open_command_expr=OPEN_POSE,
            close_command_expr=CLOSE_POSE,
        )

        # ---------- Command frame ----------
        # EE body used by object pose command (for reach reward/orientation).
        # With the hand mounted on UR5, wrist_3_link is a good base; you can
        # change this to a palm/TCP link if you have one.
        self.commands.object_pose.body_name = "wrist_3_link"

        # ---------- Object ----------
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # ---------- (Optional) EE frame visualizer ----------
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",  # URDF base link name
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
                    name="end_effector",
                    # if you have a palm/TCP offset, put it here, e.g. [0, 0, 0.10]
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )


@configclass
class UR5AdaptHandCubeLiftEnvCfg_PLAY(UR5AdaptHandCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
