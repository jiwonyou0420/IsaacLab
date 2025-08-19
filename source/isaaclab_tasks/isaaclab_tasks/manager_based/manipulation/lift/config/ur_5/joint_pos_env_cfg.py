# ur5_cube_lift_env_cfg.py
# Copyright (c) 2022-2025
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

# Markers for debugging ee frame (optional)
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

# >>>>>> Import your combined UR5 + XArm gripper articulation cfg <<<<<<
# If you saved it elsewhere, change the import accordingly.
from isaaclab_assets.robots.universal_robots import UR5_XARM_GRIPPER_CFG, UR10_ROBOTIQ_2F_140_CFG, UR5_ROBOTIQ_CFG  # isort: skip

@configclass
class UR5RobotiqCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # parent init
        super().__post_init__()

        # ---------- Robot ----------
        # Mount UR5 + XArm gripper under the env
        self.scene.robot = UR5_ROBOTIQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Actions:
        #  - 6 arm joints (explicit list to avoid accidental captures)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        #  - 1 gripper scalar mapped to multiple joints (tune targets if your USD differs)
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                # if your mechanism uses only a single drive joint, you can list just that
                # "left_finger_joint",
                # "right_finger_joint",
                # "left_inner_knuckle_joint",
                # "right_inner_knuckle_joint",
                # "right_outer_knuckle_joint",
                "finger_joint",  # include if position-driven in your USD
            ],
            open_command_expr={
                # "left_finger_joint": 0.0,
                # "right_finger_joint": 0.04,
                # "left_inner_knuckle_joint": 0.0,
                # "right_inner_knuckle_joint": 0.0,
                # "right_outer_knuckle_joint": 0.0,
                "finger_joint": 0.0,
            },
            close_command_expr={
                # "left_finger_joint": 0.0,
                # "right_finger_joint": 0.0,
                # "left_inner_knuckle_joint": 0.0,
                # "right_inner_knuckle_joint": 0.0,
                # "right_outer_knuckle_joint": 0.0,
                "finger_joint": 1.0, # from GUI inspection, gripper joint angle was 0 when open and ~40deg when closed -> would be ~0.7 in radian?
            },
        )

        # Commands: set the EE body used for object pose commands.
        # For UR5 stacks this is often "tool0" (or your gripper base link). Change if needed.
        self.commands.object_pose.body_name = "robotiq_arg2f_base_link"

        # override cube goal position, just focus on random cube spawn
        self.commands.object_pose.ranges.pos_x = (0.5, 0.5)  # x-range for object pose
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)  # y-range for object pose
        self.commands.object_pose.ranges.pos_z = (0.4, 0.4)  # z-range for object pose

        # maybe add reset joint positions event
        # self.events.reset_joints = ...
        
        # ---------- Cube object ----------
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
        # extra "ur5" under "Robot" bc that's how my usd is authored
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/base_link", 
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5/ee_link/robotiq_arg2f_base_link",  # change if your EE link differs
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],  # add tip offset if you want the frame at the fingers
                    ),
                ),
            ],
        )

@configclass
class UR5RobotiqCubeLiftEnvCfg_PLAY(UR5RobotiqCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

@configclass
class UR5XarmCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # parent init
        super().__post_init__()

        # ---------- Robot ----------
        # Mount UR5 + XArm gripper under the env
        self.scene.robot = UR5_XARM_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Actions:
        #  - 6 arm joints (explicit list to avoid accidental captures)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        #  - 1 gripper scalar mapped to multiple joints (tune targets if your USD differs)
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                # if your mechanism uses only a single drive joint, you can list just that
                # "left_finger_joint",
                # "right_finger_joint",
                # "left_inner_knuckle_joint",
                # "right_inner_knuckle_joint",
                # "right_outer_knuckle_joint",
                "drive_joint",  # include if position-driven in your USD
            ],
            open_command_expr={
                # "left_finger_joint": 0.0,
                # "right_finger_joint": 0.04,
                # "left_inner_knuckle_joint": 0.0,
                # "right_inner_knuckle_joint": 0.0,
                # "right_outer_knuckle_joint": 0.0,
                "drive_joint": 0.0,
            },
            close_command_expr={
                # "left_finger_joint": 0.0,
                # "right_finger_joint": 0.0,
                # "left_inner_knuckle_joint": 0.0,
                # "right_inner_knuckle_joint": 0.0,
                # "right_outer_knuckle_joint": 0.0,
                "drive_joint": 0.9, # from GUI inspection, gripper joint angle was 0 when open and ~40deg when closed -> would be ~0.7 in radian?
            },
        )

        # Commands: set the EE body used for object pose commands.
        # For UR5 stacks this is often "tool0" (or your gripper base link). Change if needed.
        self.commands.object_pose.body_name = "xarm_gripper_base_link"

        # override cube goal position, just focus on random cube spawn
        self.commands.object_pose.ranges.pos_x = (0.5, 0.5)  # x-range for object pose
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)  # y-range for object pose
        self.commands.object_pose.ranges.pos_z = (0.4, 0.4)  # z-range for object pose

        # maybe add reset joint positions event
        # self.events.reset_joints = ...
        
        # ---------- Cube object ----------
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
        # extra "ur5" under "Robot" bc that's how my usd is authored
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/base_link", 
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5/xarm_gripper/xarm_gripper_base_link",  # change if your EE link differs
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],  # add tip offset if you want the frame at the fingers
                    ),
                ),
            ],
        )


@configclass
class UR5XarmCubeLiftEnvCfg_PLAY(UR5XarmCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

@configclass
class UR10CubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # parent init
        super().__post_init__()

        # ---------- Robot ----------
        # Mount UR10 + Robotiq 2F-140 gripper under the env
        self.scene.robot = UR10_ROBOTIQ_2F_140_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Actions:
        #  - 6 arm joints (explicit list to avoid accidental captures)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        #  - 1 gripper scalar mapped to multiple joints (tune targets if your USD differs)
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                # if your mechanism uses only a single drive joint, you can list just that
                # "left_finger_joint",
                # "right_finger_joint",
                # "left_inner_knuckle_joint",
                # "right_inner_knuckle_joint",
                # "right_outer_knuckle_joint",
                "finger_joint",  # include if position-driven in your USD
            ],
            open_command_expr={
                # "left_finger_joint": 0.0,
                # "right_finger_joint": 0.04,
                # "left_inner_knuckle_joint": 0.0,
                # "right_inner_knuckle_joint": 0.0,
                # "right_outer_knuckle_joint": 0.0,
                "finger_joint": 0.0,
            },
            close_command_expr={
                # "left_finger_joint": 0.0,
                # "right_finger_joint": 0.0,
                # "left_inner_knuckle_joint": 0.0,
                # "right_inner_knuckle_joint": 0.0,
                # "right_outer_knuckle_joint": 0.0,
                "finger_joint": 0.7, # from GUI inspection, gripper joint angle was 0 when open and ~40deg when closed -> would be ~0.7 in radian?
            },
        )

        # Commands: set the EE body used for object pose commands.
        # For UR5 stacks this is often "tool0" (or your gripper base link). Change if needed.
        self.commands.object_pose.body_name = "robotiq_arg2f_base_link"

        # override cube goal position, just focus on random cube spawn
        self.commands.object_pose.ranges.pos_x = (0.5, 0.5)  # x-range for object pose
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)  # y-range for object pose
        self.commands.object_pose.ranges.pos_z = (0.4, 0.4)  # z-range for object pose

        # maybe add reset joint positions event
        # self.events.reset_joints = ...
        
        # ---------- Cube object ----------
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
        # this one does not have additional "ur_gripper_manual" under "Robot", because USD construction is different
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/world", 
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",  # change if your EE link differs
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],  # add tip offset if you want the frame at the fingers
                    ),
                ),
            ],
        )


@configclass
class UR10CubeLiftEnvCfg_PLAY(UR10CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False