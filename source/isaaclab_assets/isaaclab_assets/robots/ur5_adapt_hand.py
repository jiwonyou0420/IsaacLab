import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Pre-defined configs
##

UR5_ADAPT_HAND_CFG = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="/home/andres/Documents/jiwon/urdf/ur5_adapt_hand_urdf/ur5_adapt_hand.urdf",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, 
                solver_position_iteration_count=8, 
                solver_velocity_iteration_count=0
            ),
            fix_base=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=800.0,
                    damping=40.0,
                ),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.712,
                "elbow_joint": 1.5,
                "wrist_1_joint": -1.5708,
                "wrist_2_joint": -1.5708,
                "wrist_3_joint": -1.5708,
                "Thumb_CMC1": 0.04,
                "Thumb_CMC2": -0.04,
                "Thumb_MCP": -0.04,
                "Thumb_IP": -0.04,
                "Index_MCP": -0.04,
                "Index_PIP": -0.04,
                "Index_DIP": -0.04,
                "Middle_MCP": -0.04,
                "Middle_PIP": -0.04,
                "Middle_DIP": -0.04,
                "Ring_MCP": -0.04,
                "Ring_PIP": -0.04,
                "Ring_DIP": -0.04,
                "Pinky_MCP": -0.04,
                "Pinky_PIP": -0.04,
                "Pinky_DIP": -0.04,
                "Index_MCP_Spread": 0.0,
                "Middle_MCP_Spread": 0.0,
                "Ring_MCP_Spread": 0.0,
                "Pinky_MCP_Spread": 0.0,
                "Wrist_Pitch": 0.0,
                "Wrist_Yaw": 0.0,
            },
        ),
        actuators={
            "ur5_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint"],
                velocity_limit_sim=100.0,
                effort_limit_sim=150.0,  
                stiffness=800.0,
                damping=40.0,
            ),
            "ur5_arm": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                velocity_limit_sim=100.0,
                effort_limit_sim=150.0, 
                stiffness=800.0,
                damping=40.0,
            ),
            "ur5_wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_[1-3]_joint"],
                velocity_limit_sim=100.0,
                effort_limit_sim=28.0,
                stiffness=800.0,
                damping=40.0,
            ),
            # Hand actuators - only for non-mimic joints
            "thumb_cmc": ImplicitActuatorCfg(
                joint_names_expr=["Thumb_CMC[1-2]"],
                effort_limit_sim=10.0,
                velocity_limit_sim=2.5,
                stiffness=2e3,
                damping=1e2,
            ),
            "thumb_mcp": ImplicitActuatorCfg(
                joint_names_expr=["Thumb_MCP", "Thumb_IP"],
                effort_limit_sim=10.0,
                velocity_limit_sim=2.5,
                stiffness=2e3,
                damping=1e2,
            ),
            "finger_spread": ImplicitActuatorCfg(
                joint_names_expr=["Pinky_MCP_Spread", "Index_MCP_Spread", "Middle_MCP_Spread", "Ring_MCP_Spread"], 
                effort_limit_sim=10.0,
                velocity_limit_sim=2.5,
                stiffness=2e3,
                damping=1e2,
            ),
            "finger_mcp": ImplicitActuatorCfg(
                joint_names_expr=["Pinky_MCP", "Ring_MCP", "Middle_MCP", "Index_MCP"],
                effort_limit_sim=10.0,
                velocity_limit_sim=2.5,
                stiffness=2e3,
                damping=1e2,
            ),
            "finger_pip": ImplicitActuatorCfg(
                joint_names_expr=["Pinky_PIP", "Ring_PIP", "Middle_PIP", "Index_PIP"],
                effort_limit_sim=10.0,
                velocity_limit_sim=2.5,
                stiffness=2e3,
                damping=1e2,
            ),
            "finger_dip": ImplicitActuatorCfg(
                joint_names_expr=["Pinky_DIP", "Ring_DIP", "Middle_DIP", "Index_DIP"],
                effort_limit_sim=10.0,
                velocity_limit_sim=2.5,
                stiffness=2e3,
                damping=1e2,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["Wrist_Pitch", "Wrist_Yaw"],
                effort_limit_sim=10.0,
                velocity_limit_sim=2.5,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )


