# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##


UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
"""Configuration of UR-10 arm using implicit actuator models."""


# UR5 with xarm gripper
# UR5 with XArm gripper (single articulation)
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

UR5_XARM_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # your mounted robot+gripper USD
        usd_path="/home/andres/Documents/jiwon/isaacsim_scenes/ur5_with_xarm_gripper.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),

    # -------- Default joint state --------
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # UR5 arm (typical neutral)
            "shoulder_pan_joint":   0.0,
            "shoulder_lift_joint": -1.2,   # ~-98 deg
            "elbow_joint":          0.9,   # ~+98 deg
            "wrist_1_joint":        -1.4,
            "wrist_2_joint":        -1.5,
            "wrist_3_joint":        1.9,

            # XArm gripper (example names; change to match your USD)
            # Choose an "open" default that won’t collide:
            "drive_joint":                 0.0,   # often the driven transmission input
            "left_finger_joint":           0.0,  # meters or radians depending on joint type
            "right_finger_joint":          0.0,
            "left_inner_knuckle_joint":    0.0,
            "right_inner_knuckle_joint":   0.0,
            "right_outer_knuckle_joint":   0.0,
        },
        # If you want initial joint velocities, add "joint_vel={...}" here
    ),

    # -------- Actuator groups (PD gains / limits) --------
    actuators={
        # UR5 arm: one actuator group for 6 arm joints
        "ur5_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            # ballpark limits/gains; tune as needed
            velocity_limit_sim=3.2,     # URs are ~3.14–3.2 rad/s
            effort_limit_sim=150.0,     # per-URDF upper bound (arm joints)
            stiffness=600.0,            # PD Kp
            damping=30.0,               # PD Kd
        ),

        # XArm gripper: group all mechanical joints of the gripper
        # (You will still expose ONE "gripper" action later in your env cfg.)
        "xarm_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "drive_joint",
                # "left_finger_joint",
                # "right_finger_joint",
                # "left_inner_knuckle_joint",
                # "right_inner_knuckle_joint",
                # "right_outer_knuckle_joint",
            ],
            velocity_limit_sim=0.5,     # grippers are slow
            effort_limit_sim=200.0,     # give it some authority
            stiffness=2000.0,           # stiffer to hold against contact
            damping=100.0,
        ),
    },

    # Keep position limits active as authored in USD
    soft_joint_pos_limit_factor=1.0,
)
