"""Optional actor-only checkpoint loading for H2 tracking."""

import torch
from rsl_rl.env import VecEnv

from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner


class H2TrackingOnPolicyRunner(MotionTrackingOnPolicyRunner):
  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device, registry_name)
    checkpoint = train_cfg.get("actor_checkpoint")
    if checkpoint and not train_cfg.get("resume", False):
      state = torch.load(checkpoint, map_location=device, weights_only=False)
      self.alg.get_policy().load_state_dict(state["actor_state_dict"], strict=True)
      print(f"[H2] Loaded actor only: {checkpoint}; fresh critic/optimizer/iteration")
