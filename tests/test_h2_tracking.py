"""Regression tests for the closed-loop H2 policy contract."""

import os
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots.unitree_h2.h2_constants import (
  H2_ACTUATED_JOINTS,
  H2_JOINT_NAMES,
  H2_XML,
  get_h2_robot_cfg,
  get_spec,
)
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.config.h2.env_cfgs import unitree_h2_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.h2.mdp import (
  head_joint_position_error_exp,
  validate_motion,
)


@pytest.fixture(scope="module")
def h2_model() -> mujoco.MjModel:
  """Compile the real H2 asset without constructing a GPU environment."""
  path = Path(os.environ.get("MJLAB_H2_XML", str(H2_XML))).expanduser()
  if not path.is_file():
    pytest.skip("Provide external H2 assets through MJLAB_H2_XML")
  return Entity(get_h2_robot_cfg()).spec.compile()


def test_h2_joint_and_motor_order(h2_model: mujoco.MjModel) -> None:
  """Preserve all passive joints and the sim2sim actuator ordering."""
  assert (h2_model.nq, h2_model.nv, h2_model.nu) == (44, 43, 31)
  assert tuple(h2_model.joint(i).name for i in range(1, 38)) == H2_JOINT_NAMES
  assert (
    tuple(h2_model.joint(int(j)).name for j in h2_model.actuator_trnid[:, 0])
    == H2_ACTUATED_JOINTS
  )


def test_h2_unlimited_positions_but_limited_effort(h2_model: mujoco.MjModel) -> None:
  """Remove position clipping without removing physical effort saturation."""
  assert not h2_model.jnt_limited.any()
  assert not h2_model.actuator_ctrllimited.any()
  assert h2_model.actuator_forcelimited.all()
  assert (h2_model.actuator_forcerange[:, 1] > 0).all()
  np.testing.assert_allclose(
    h2_model.actuator_gainprm[:, 0], -h2_model.actuator_biasprm[:, 1]
  )
  np.testing.assert_allclose(h2_model.dof_damping[6:], 0.05)
  assert h2_model.geom("pelvis_collision").conaffinity == 1
  assert h2_model.geom("left_foot_front_collision").condim == 3


def test_h2_keep_original_tendon_equalities(h2_model: mujoco.MjModel) -> None:
  """Retain every rod, its reference length, and its XML solver parameters."""
  path = Path(os.environ.get("MJLAB_H2_XML", str(H2_XML))).expanduser()
  original = mujoco.MjModel.from_xml_path(str(path))
  assert h2_model.neq == h2_model.ntendon == 6
  assert (h2_model.eq_type == mujoco.mjtEq.mjEQ_TENDON).all()
  for name in ("eq_data", "eq_solref", "eq_solimp", "tendon_length0"):
    np.testing.assert_allclose(getattr(h2_model, name), getattr(original, name))
  np.testing.assert_allclose(h2_model.eq_solref[:, 0], 0.005)


def test_h2_play_and_head_configuration() -> None:
  """Keep the actor contract and make head tracking explicit in both modes."""
  for play in (False, True):
    cfg = unitree_h2_flat_tracking_env_cfg(play=play)
    assert cfg.scene.num_envs == (1 if play else 8196)
    assert tuple(cfg.observations["actor"].terms) == (
      "command",
      "motion_anchor_pos_b",
      "motion_anchor_ori_b",
      "base_lin_vel",
      "base_ang_vel",
      "joint_pos",
      "joint_vel",
      "actions",
    )
    assert all(
      name in cfg.rewards
      for name in ("head_joint_pos", "head_body_ori", "head_body_pos")
    )
    assert "joint_limit" not in cfg.rewards
    assert cfg.sim.mujoco.solver == "newton"
    assert cfg.sim.mujoco.timestep * cfg.decimation == 0.02
    if play:
      assert not cfg.events


def test_head_joint_reward_responds_to_head_only_error() -> None:
  """Penalize head error even when all other joints track perfectly."""
  command = SimpleNamespace(
    robot_joint_pos=torch.zeros(2, 37), joint_pos=torch.zeros(2, 37)
  )
  command.robot_joint_pos[1, 19] = 0.4
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
  asset = SceneEntityCfg("robot", joint_ids=[19, 20])
  rewards = head_joint_position_error_exp(env, "motion", 0.2, asset)  # type: ignore[arg-type]
  torch.testing.assert_close(
    rewards, torch.tensor([1.0, np.exp(-2)], dtype=torch.float32)
  )


def test_h2_motion_validation(tmp_path: Path) -> None:
  """Reject wrong FPS, body order dimensions, and non-finite reference data."""
  data = {
    "fps": np.array(50.0),
    "joint_pos": np.zeros((2, 37)),
    "joint_vel": np.zeros((2, 37)),
    "body_pos_w": np.zeros((2, 38, 3)),
    "body_quat_w": np.tile([1.0, 0, 0, 0], (2, 38, 1)),
    "body_lin_vel_w": np.zeros((2, 38, 3)),
    "body_ang_vel_w": np.zeros((2, 38, 3)),
  }
  path = tmp_path / "motion.npz"
  np.savez(path, allow_pickle=False, **data)
  validate_motion(str(path), 0.02)
  with pytest.raises(ValueError, match="FPS"):
    validate_motion(str(path), 0.01)
  data["joint_pos"][0, 0] = np.nan
  np.savez(path, allow_pickle=False, **data)
  with pytest.raises(ValueError, match="joint_pos"):
    validate_motion(str(path), 0.02)


@pytest.mark.parametrize("override", [False, True])
def test_h2_external_xml(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: bool
) -> None:
  """Load external assets at use time without modifying the source XML."""
  monkeypatch.chdir(tmp_path)
  monkeypatch.delenv("MJLAB_H2_XML", raising=False)
  path = tmp_path / ("custom.xml" if override else "h2_description/H2loop_complete.xml")
  path.parent.mkdir(parents=True, exist_ok=True)
  xml = """<mujoco><worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body><joint name="hinge" limited="true" range="-10 10"/>
      <geom type="sphere" size=".1"/></body>
    </worldbody><actuator><motor joint="hinge"/></actuator></mujoco>"""
  path.write_text(xml)
  if override:
    monkeypatch.setenv("MJLAB_H2_XML", str(path))
  model = get_spec().compile()
  assert model.njnt == 1
  assert not model.jnt_limited.any()
  assert model.nu == 0
  assert model.ngeom == 1
  assert path.read_text() == xml


def test_h2_missing_xml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Explain how to supply missing assets without blocking task discovery."""
  monkeypatch.setenv("MJLAB_H2_XML", str(tmp_path / "missing.xml"))
  unitree_h2_flat_tracking_env_cfg(play=True)
  with pytest.raises(FileNotFoundError, match="MJLAB_H2_XML"):
    get_spec()
