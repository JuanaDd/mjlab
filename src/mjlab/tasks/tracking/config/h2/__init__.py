from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_h2_flat_tracking_env_cfg
from .rl_cfg import unitree_h2_tracking_ppo_runner_cfg
from .runner import H2TrackingOnPolicyRunner

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-H2",
  env_cfg=unitree_h2_flat_tracking_env_cfg(),
  play_env_cfg=unitree_h2_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_h2_tracking_ppo_runner_cfg(),
  runner_cls=H2TrackingOnPolicyRunner,
)
