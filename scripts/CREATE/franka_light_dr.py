# ---- light randomizer (function-style, for EventTermCfg(func=..., mode="reset")) ----
from __future__ import annotations
from typing import Iterable, Tuple, Sequence, Literal
import math
import random
import torch

from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg
def _env_light_path(env, env_id: int, name: str = "Light") -> str:
    """Returns light prim path for a given env index."""
    # Typically: /World/envs/env_0/Light — matches Franka Lift scene default.
    base = env.scene.envs_prim_path  # e.g., "/World/envs"
    return f"{base}/env_{env_id}/{name}"

def _ensure_parent_xform(stage: Usd.Stage, prim_path: str):
    parent_path = Sdf.Path(prim_path).GetParentPath()
    if not stage.GetPrimAtPath(parent_path):
        UsdGeom.Xform.Define(stage, parent_path.pathString)

def _delete_if_exists(stage: Usd.Stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if prim:
        stage.RemovePrim(prim_path)

def _set_xform(prim: Usd.Prim, pos: Tuple[float,float,float], rot_euler_deg: Tuple[float,float,float] = (0,0,0)):
    xf = UsdGeom.Xformable(prim)
    # clear any existing ops to keep it simple/consistent
    for op in xf.GetOrderedXformOps():
        xf.RemoveXformOp(op)
    # translate
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    # rotate XYZ in degrees (helps to point distant light)
    rx, ry, rz = rot_euler_deg
    xf.AddRotateXYZOp().Set(Gf.Vec3f(rx, ry, rz))

def _look_down_towards(target_w: Sequence[float], from_pos_w: Sequence[float]) -> Tuple[float,float,float]:
    """Returns simple Euler XYZ (deg) to roughly point -Z axis towards target (for DistantLight)."""
    # Vector pointing from light to target
    dx = target_w[0] - from_pos_w[0]
    dy = target_w[1] - from_pos_w[1]
    dz = target_w[2] - from_pos_w[2]
    # Aim the light "forward" along -Z, so we tilt such that -Z aligns to this vector.
    # We can approximate with yaw (around Z) then pitch (around Y).
    yaw = math.degrees(math.atan2(dy, dx))                    # spin around Z
    horiz = math.hypot(dx, dy)
    pitch = math.degrees(math.atan2(horiz, -dz))              # negative Z is forward
    # roll = 0
    return (0.0, pitch, yaw)

def randomize_light(
    env,
    env_ids: torch.Tensor | None,
    light_name: str = "Light",
    # which types to sample from
    light_types: Iterable[Literal["DomeLight","SphereLight","DistantLight"]] = ("DomeLight","SphereLight","DistantLight"),
    # intensity ranges per type
    intensity_range_common: Tuple[float,float] = (500.0, 5000.0),
    dome_intensity_range: Tuple[float,float] | None = None,
    # sphere placement (relative to env origin)
    sphere_height_range: Tuple[float,float] = (0.9, 1.6),
    sphere_radius_xy_range: Tuple[float,float] = (0.3, 1.0),
    # distant light direction anchor (table center-ish in robot frame)
    target_offset_w: Tuple[float,float,float] = (0.6, 0.0, 0.2),
    # optionally vary exposure (applied in addition to intensity)
    exposure_range: Tuple[float,float] | None = None,
    # optionally vary color temperature for dome
    vary_color_temperature: bool = False,
    color_temp_range: Tuple[float,float] = (3500.0, 8000.0),
):
    """
    Randomize one light *per env* at reset. Safe to use as EventTermCfg(func=..., mode="reset").
    - Replaces /World/envs/env_i/<light_name> with a randomly chosen USDLux light type.
    - Positions sphere lights above workspace; orients distant light to point to table area.
    - Randomizes intensity (and optional exposure, dome color-temp).
    """
    stage = sim_utils.get_current_stage()
    device = env.device

    # resolve env ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # precompute per-type intensity ranges
    dome_lo, dome_hi = dome_intensity_range if dome_intensity_range is not None else intensity_range_common
    com_lo, com_hi = intensity_range_common

    # target location (world): approximate from env origin + offset
    env_origins = env.scene.env_origins.cpu().numpy()

    # loop each env id and (re)spawn its light
    for i in env_ids.tolist():
        prim_path = _env_light_path(env, i, light_name)
        _ensure_parent_xform(stage, prim_path)
        _delete_if_exists(stage, prim_path)

        # pick light type
        ltype = random.choice(tuple(light_types))

        # pick intensity & exposure
        if ltype == "DomeLight":
            intensity = random.uniform(dome_lo, dome_hi)
        else:
            intensity = random.uniform(com_lo, com_hi)
        exposure = random.uniform(*exposure_range) if exposure_range is not None else 0.0

        if ltype == "DomeLight":
            light = UsdLux.DomeLight.Define(stage, prim_path)
            # place at origin (transform doesn’t matter much for dome)
            _set_xform(light.GetPrim(), pos=(0.0, 0.0, 0.0), rot_euler_deg=(0.0, 0.0, 0.0))
            # set properties
            light.CreateIntensityAttr(intensity)
            light.CreateExposureAttr(exposure)
            # color temperature?
            if vary_color_temperature:
                light.CreateEnableColorTemperatureAttr(True)
                light.CreateColorTemperatureAttr(random.uniform(*color_temp_range))
            else:
                light.CreateEnableColorTemperatureAttr(False)

        elif ltype == "SphereLight":
            light = UsdLux.SphereLight.Define(stage, prim_path)
            # sample a ring above the table around env origin
            origin = env_origins[i]
            r = random.uniform(*sphere_radius_xy_range)
            theta = random.uniform(-math.pi, math.pi)
            x = origin[0] + r * math.cos(theta)
            y = origin[1] + r * math.sin(theta)
            z = origin[2] + random.uniform(*sphere_height_range)
            _set_xform(light.GetPrim(), pos=(x, y, z), rot_euler_deg=(0.0, 0.0, 0.0))
            # properties
            light.CreateIntensityAttr(intensity)
            light.CreateExposureAttr(exposure)
            # small radius so shadows are fairly sharp
            light.CreateRadiusAttr(0.05)
            light.CreateTreatAsPointAttr(False)

        elif ltype == "DistantLight":
            light = UsdLux.DistantLight.Define(stage, prim_path)
            origin = env_origins[i]
            # put “position” slightly above origin (transform controls direction; position is arbitrary)
            pos = (origin[0], origin[1], origin[2] + 1.5)
            target = (origin[0] + target_offset_w[0],
                      origin[1] + target_offset_w[1],
                      origin[2] + target_offset_w[2])
            euler = _look_down_towards(target, pos)
            _set_xform(light.GetPrim(), pos=pos, rot_euler_deg=euler)
            # properties
            light.CreateIntensityAttr(intensity)
            light.CreateExposureAttr(exposure)
            # sun-ish angular size (soft shadows): 0.3–2.0 deg
            light.CreateAngleAttr(random.uniform(0.3, 2.0))

        else:
            # fallback: SphereLight
            light = UsdLux.SphereLight.Define(stage, prim_path)
            origin = env_origins[i]
            _set_xform(light.GetPrim(), pos=(origin[0], origin[1], origin[2] + 1.2), rot_euler_deg=(0,0,0))
            light.CreateIntensityAttr(intensity)
            light.CreateExposureAttr(exposure)
            light.CreateRadiusAttr(0.05)

def apply_light_randomization(env_cfg):
    env_cfg.events.random_light = EventTermCfg(
    func=randomize_light,
    mode="reset",
    params={
        "light_name": "Light",                     # matches your scene’s light prim name
        "light_types": ("DomeLight","SphereLight","DistantLight"),
        "intensity_range_common": (800.0, 6000.0),
        "dome_intensity_range": (0.5, 3.0),       # domes are “bright” by design; this is multiplied further
        "exposure_range": (-2.0, 3.0),
        "sphere_height_range": (1.0, 1.8),
        "sphere_radius_xy_range": (0.3, 1.2),
        "target_offset_w": (0.6, 0.0, 0.25),      # where we want distant light to “look”
        "vary_color_temperature": True,
        "color_temp_range": (3000.0, 8000.0),
    },
)
    
