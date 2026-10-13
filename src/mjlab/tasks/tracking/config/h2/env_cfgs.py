"""H2 closed-loop mimic with a sim2sim-compatible actor and head rewards."""

from mjlab.asset_zoo.robots.unitree_h2.h2_constants import (
  H2_ACTION_SCALE,
  H2_FOOT_GEOMS,
  H2_HEAD_BODIES,
  H2_TRACKED_BODIES,
  get_h2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from .mdp import H2MotionCommandCfg, head_joint_position_error_exp


def unitree_h2_flat_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build a 50 Hz mimic task while preserving the XML tendon equalities."""
  cfg = make_tracking_env_cfg()
  cfg.scene.num_envs = 1 if play else 8196
  cfg.scene.entities = {"robot": get_h2_robot_cfg()}
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  )
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = H2_ACTION_SCALE.copy()
  original = cfg.commands["motion"]
  assert isinstance(original, MotionCommandCfg)
  motion = H2MotionCommandCfg(**vars(original))
  motion.anchor_body_name = "torso_link"
  motion.body_names = H2_TRACKED_BODIES + H2_HEAD_BODIES
  # Independent noise on linkage input/output coordinates breaks rod closure.
  motion.joint_position_range = (0.0, 0.0)
  cfg.commands["motion"] = motion
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = H2_FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )
  # Unrestricted positions are intentional; retain actuator force saturation.
  cfg.rewards.pop("joint_limit")
  cfg.rewards.pop("self_collisions")
  cfg.rewards["head_joint_pos"] = RewardTermCfg(
    func=head_joint_position_error_exp,
    weight=0.5,
    params={
      "command_name": "motion",
      "std": 0.2,
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=("head_pitch_joint", "head_yaw_joint")
      ),
    },
  )
  cfg.rewards["head_body_ori"] = RewardTermCfg(
    func=mdp.motion_relative_body_orientation_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.2, "body_names": H2_HEAD_BODIES},
  )
  cfg.rewards["head_body_pos"] = RewardTermCfg(
    func=mdp.motion_relative_body_position_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.1, "body_names": H2_HEAD_BODIES},
  )
  cfg.decimation = 4
  cfg.sim.mujoco.timestep = 0.005
  cfg.sim.mujoco.integrator = "implicitfast"
  cfg.sim.mujoco.solver = "newton"
  cfg.sim.mujoco.iterations = 100
  cfg.sim.mujoco.ls_iterations = 50
  cfg.sim.mujoco.tolerance = 1e-8
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 512
  cfg.viewer.body_name = "torso_link"
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.clear()
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.sampling_mode = "start"
  return cfg
