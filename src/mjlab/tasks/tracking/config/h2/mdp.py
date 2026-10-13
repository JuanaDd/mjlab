"""H2 motion validation and an explicit, undiluted head tracking reward."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from mjlab.asset_zoo.robots.unitree_h2.h2_constants import H2_JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg


def validate_motion(path: str, step_dt: float) -> None:
  """Reject incompatible motion files instead of silently changing replay speed."""
  with np.load(path, allow_pickle=False) as data:
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if not np.isclose(fps * step_dt, 1.0):
      raise ValueError(f"Motion FPS {fps} does not match control dt {step_dt}")
    frames = data["joint_pos"].shape[0]
    if frames < 2:
      raise ValueError("H2 motion needs at least two frames")
    for key, tail in {
      "joint_pos": (37,),
      "joint_vel": (37,),
      "body_pos_w": (38, 3),
      "body_quat_w": (38, 4),
      "body_lin_vel_w": (38, 3),
      "body_ang_vel_w": (38, 3),
    }.items():
      if data[key].shape != (frames, *tail) or not np.isfinite(data[key]).all():
        raise ValueError(f"Invalid H2 motion array: {key} {data[key].shape}")
    if not np.allclose(np.linalg.norm(data["body_quat_w"], axis=-1), 1, atol=1e-3):
      raise ValueError("Motion quaternions must be normalized, in wxyz order")


@dataclass(kw_only=True)
class H2MotionCommandCfg(MotionCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    """Check the 37-joint motion contract before constructing the mimic command."""
    validate_motion(self.motion_file, env.step_dt)
    if tuple(env.scene[self.entity_name].joint_names) != H2_JOINT_NAMES:
      raise ValueError("H2 joint order does not match the policy interface")
    return MotionCommand(self, env)


def head_joint_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Track both head joints without averaging their error across the whole body."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = (
    (
      command.robot_joint_pos[:, asset_cfg.joint_ids]
      - command.joint_pos[:, asset_cfg.joint_ids]
    )
    .square()
    .mean(dim=-1)
  )
  return torch.exp(-error / std**2)
