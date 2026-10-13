"""PPO configuration for the H2 mimic policy interface."""

from dataclasses import dataclass

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


@dataclass
class H2TrackingRunnerCfg(RslRlOnPolicyRunnerCfg):
  actor_checkpoint: str | None = None
  """Optional actor-only warm start; critic and optimizer start fresh."""


def unitree_h2_tracking_ppo_runner_cfg() -> H2TrackingRunnerCfg:
  """Configure the 194-input, 31-output ELU policy and mimic PPO defaults."""
  base = unitree_g1_tracking_ppo_runner_cfg()
  return H2TrackingRunnerCfg(
    actor=base.actor,
    critic=base.critic,
    algorithm=base.algorithm,
    experiment_name="h2_closeloop_mimic",
    num_steps_per_env=24,
    max_iterations=30_000,
    save_interval=250,
    logger="tensorboard",
    upload_model=False,
  )
