# adds wrist cam and birdeye cam to FrankaCubeLiftEnv
# a minor modification for early episode termination if cube is stable around goal
# TODO: add contact sensors, make similar envs for UR5/hand, etc


from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab_tasks.manager_based.manipulation.lift.mdp import object_at_goal_stable

@configclass
class FrankaCubeLiftCamEnvCfg(FrankaCubeLiftEnvCfg):
    """Franka Lift-Cube env + two extra cameras (wrist + bird's-eye)."""

    def __post_init__(self):
        super().__post_init__()
        # --- Wrist camera mounted on the end-effector (panda_hand) ---
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            width=1280,
            height=720,
            update_period=0.0,  # every sim step
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.03, 1.0e5),
            ),
            # Start with a mild forward offset; tune in the GUI.
            # Set convention to "ros" so XYZ axes match the usual REP-103.
            offset=CameraCfg.OffsetCfg(
                pos=(0.1, 0, -0.1), rot=(-0.10452, -0.69934, -0.69934, -0.10452), convention="opengl" 
            ),
            debug_vis=True,  # shows frustum so you can tune interactively
        )

        # --- Bird's-eye camera in world ---
        self.scene.bird_cam = CameraCfg(
            prim_path="/World/BirdEyeCamera",
            width=1280,
            height=720,
            update_period=0.0,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.9, 0.0, 1.3),                # above the table; adjust in GUI
                rot=(0.64086, 0.29884, 0.29884, 0.64086),  # identity; tilt in GUI
                convention="opengl",
            ),
            debug_vis=True,
        )

        self.commands.object_pose.debug_vis = False  # disable debug vis for goal position

        # episode will terminate when the object is at the goal and stable
        self.terminations.object_stable_goal = DoneTerm(
            func=object_at_goal_stable,
            params=dict(
                pos_threshold=0.1,
                lin_vel_threshold=0.08,
                ang_vel_threshold=0.5,
                hold_time_s=0.5,    # tune: e.g., 0.3–1.0 s
            ),
            time_out=True
        )
