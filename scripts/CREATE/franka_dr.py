# scripts/CREATE/franka_dr.py
# Usage:
#   from franka_dr import apply_franka_cam_dr
#   env_cfg = FrankaCubeLiftCamEnvCfg()
#   apply_franka_cam_dr(env_cfg, cube_texture_paths=[...], table_texture_paths=[...])

from __future__ import annotations
from typing import Sequence

from isaaclab.managers import EventTermCfg, SceneEntityCfg
# Built-in event randomization functions (mass, materials, textures, etc.)
from isaaclab.envs.mdp import events as events_mdp
import os, glob
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# --- Cube textures (Blocks) ---
cube_textures = sorted(
    glob.glob(os.path.join(ISAAC_NUCLEUS_DIR, "Props/Blocks/Materials/Textures/*.png"))
) 

# --- Table textures (SeattleLabTable) ---
table_textures = sorted(
    glob.glob(os.path.join(ISAAC_NUCLEUS_DIR, "Props/Mounts/SeattleLabTable/Materials/Textures/*.png"))
)


def apply_franka_cam_dr(
    env_cfg,
    # ---- Cube mass & friction ----
    cube_mass_scale_range: tuple[float, float] = (0.5, 1.5),        # multiplicative scale on default mass
    cube_static_friction_range: tuple[float, float] = (0.5, 1.3),
    cube_dynamic_friction_range: tuple[float, float] = (0.4, 1.0),
    cube_restitution_range: tuple[float, float] = (0.0, 0.05),
    cube_material_buckets: int = 64,  # number of material presets to sample from

    # ---- Optional texture randomization ----
    cube_texture_paths: Sequence[str] | None = cube_textures,   # e.g., list of USD texture files/images
    table_texture_paths: Sequence[str] | None = table_textures,  # same as above
    texture_rotation_range: tuple[float, float] = (0.0, 3.14159),  # radians

) -> None:
    """
    Attach domain randomization terms to a Franka*Lift* env cfg.

    Notes:
    - 'object' is assumed to be the cube entity name in the scene.
    - 'table' is assumed to be the table entity name in the scene.
    - Texture randomization requires scene.replicate_physics = False (set here).
    - All terms run on 'reset' (i.e., each episode reset).
    """
    # Ensure the events container exists
    if getattr(env_cfg, "events", None) is None:
        # Older envs: some configs keep EventCfg under env_cfg.events
        # If not present, create a simple namespace-like holder.
        class _Events: pass
        env_cfg.events = _Events()

    # -------- Materials / Mass on the cube ("object") --------
    # Mass randomization (scales default masses). Uses CPU-safe path; best applied at reset.
    env_cfg.events.cube_mass = EventTermCfg(
        func=events_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "mass_distribution_params": cube_mass_scale_range,
            "operation": "scale",          # multiplicative scale
            "distribution": "log_uniform", # common choice for multiplicative mass changes
            "recompute_inertia": True,
        },
    )

    # Physics material randomization: static/dynamic friction + restitution
    env_cfg.events.cube_material = EventTermCfg(
        func=events_mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": cube_static_friction_range,
            "dynamic_friction_range": cube_dynamic_friction_range,
            "restitution_range": cube_restitution_range,
            "num_buckets": cube_material_buckets,
            "make_consistent": True,   # ensure dynamic <= static
        },
    )

    # -------- Optional texture randomization (requires Replicator; no physics replication) --------
    # Texture randomization needs per-prim material binding; Isaac Lab’s helper handles it.
    # It requires env_cfg.scene.replicate_physics = False (USD-level randomization per instance).
    # We set that here if textures are requested.
    want_textures = bool(cube_texture_paths) or bool(table_texture_paths)
    if want_textures:
        env_cfg.scene.replicate_physics = False

    if cube_texture_paths:
        env_cfg.events.cube_texture = EventTermCfg(
            func=events_mdp.randomize_visual_texture_material,
            mode="reset",
            params={
                "event_name": "cube_texture_reset",             # internal label for Replicator trigger (OK if unused)
                "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                "texture_paths": list(cube_texture_paths),
                "texture_rotation": texture_rotation_range,     # radians
            },
        )

    if table_texture_paths:
        # Table visuals can vary by asset; we target all meshes under the table prim
        env_cfg.events.table_texture = EventTermCfg(
            func=events_mdp.randomize_visual_texture_material,
            mode="reset",
            params={
                "event_name": "table_texture_reset",
                "asset_cfg": SceneEntityCfg("table", body_names=".*"),
                "texture_paths": list(table_texture_paths),
                "texture_rotation": texture_rotation_range,
            },
        )

    # No lighting randomization here: there is no built-in lighting DR term in the events MDP utilities.
    # If you need lights, we can add a small Replicator script later that randomizes light prims by path.

    # That’s it — the env will apply these on every env.reset()

