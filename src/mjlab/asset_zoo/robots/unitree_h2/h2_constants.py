"""Closed-loop H2 asset and the 37-joint, 31-action policy interface."""

import math
import os
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

H2_XML = Path("h2_description/H2loop_complete.xml")

H2_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_roll_joint",
  "left_ankle_pitch_joint",
  "left_ankle_motor_joint",
  "left_knee_motor_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_roll_joint",
  "right_ankle_pitch_joint",
  "right_ankle_motor_joint",
  "right_knee_motor_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "head_pitch_joint",
  "head_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
  "torso_constraint_R_joint",
  "torso_constraint_L_joint",
)
H2_PASSIVE_JOINTS = (
  "left_knee_joint",
  "right_knee_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
)
H2_ACTUATED_JOINTS = tuple(n for n in H2_JOINT_NAMES if n not in H2_PASSIVE_JOINTS)
H2_TRACKED_BODIES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
H2_HEAD_BODIES = ("head_pitch_link", "head_yaw_link")
H2_FOOT_GEOMS = r"^(left|right)_foot_(front|rear|inner|outer)_collision$"
H2_DEFAULT_JOINT_POS = {
  "left_shoulder_pitch_joint": 0.297,
  "right_shoulder_pitch_joint": 0.297,
  "left_shoulder_roll_joint": 0.281,
  "right_shoulder_roll_joint": -0.281,
  "left_elbow_joint": 0.718,
  "right_elbow_joint": 0.718,
}

_OMEGA = 2 * math.pi * 10
_GROUPS = (
  (r".*_hip_(pitch|roll|yaw)_joint|.*_knee_motor_joint", 0.160478, 254.0),
  (r".*_ankle_(roll|motor)_joint", 0.0190185, 53.7),
  (
    r"waist_yaw_joint|.*_shoulder_.*_joint|.*_elbow_joint|torso_constraint_[LR]_joint",
    0.0251019,
    111.0,
  ),
  (r"head_.*_joint", 0.01, 50.0),
  (r".*_wrist_roll_joint", 0.01, 60.0),
  (r".*_wrist_(pitch|yaw)_joint", 0.01, 10.0),
)
H2_ACTION_SCALE = {
  pattern: 0.25 * effort / (armature * _OMEGA**2)
  for pattern, armature, effort in _GROUPS
}


def get_spec() -> mujoco.MjSpec:
  """Load the original six-tendon mechanism without position/command clipping."""
  # Defer external asset resolution so task discovery needs no robot files.
  xml_path = Path(os.environ.get("MJLAB_H2_XML", str(H2_XML))).expanduser()
  if not xml_path.is_file():
    raise FileNotFoundError(
      f"H2 XML not found: {xml_path}. Set MJLAB_H2_XML or place the XML and "
      "its meshes in h2_description/; see docs/h2.md."
    )
  spec = mujoco.MjSpec.from_file(str(xml_path))
  # Recreate motors as position servos; keep XML torque bounds and rod equalities.
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for key in list(spec.keys):
    spec.delete(key)
  for joint in spec.joints:
    if joint.type != mujoco.mjtJoint.mjJNT_FREE:
      joint.limited = False
  # The task owns the terrain, not the robot asset.
  spec.delete(spec.geom("floor"))
  return spec


def get_h2_robot_cfg() -> EntityCfg:
  """Return fresh H2 configuration with 31 PD motors and six passive outputs."""
  return EntityCfg(
    spec_fn=get_spec,
    sort_actuators=True,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 1.05),
      joint_pos=H2_DEFAULT_JOINT_POS.copy(),
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=tuple(
        BuiltinPositionActuatorCfg(
          target_names_expr=(pattern,),
          stiffness=armature * _OMEGA**2,
          damping=4 * armature * _OMEGA,
          effort_limit=effort,
          armature=armature,
          viscous_damping=0.05,
        )
        for pattern, armature, effort in _GROUPS
      ),
      soft_joint_pos_limit_factor=1.0,
    ),
    collisions=(
      CollisionCfg(
        geom_names_expr=(r".*_collision",),
        contype=0,
        conaffinity=1,
        condim={H2_FOOT_GEOMS: 3, ".*": 1},
        priority={H2_FOOT_GEOMS: 1, ".*": 0},
        friction={H2_FOOT_GEOMS: (0.6,)},
      ),
    ),
  )
