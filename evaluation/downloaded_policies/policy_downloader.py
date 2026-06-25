"""
Centralized W&B policy downloader for the UR3 pick task.

Single source of truth for fetching a trained PPO policy from W&B into a
load_policy_fn-ready local cache keyed by run_id:

    evaluation/downloaded_policies/{run_id}/
    ├── params.msgpack     # copied from W&B
    └── metadata.json      # env_name, obs_dim, action_dim, network_factory, ...

Fetch source: prefers the run's Files tab `trained_policy/` folder (written by
robots/UR3e/run_experiment.py since the centralization change); falls back to
the run's `model` artifact for older runs that predate it.

Dependency-light on purpose: only wandb + the mujoco_playground registry (for
fallback obs/action dims). No robot/RTDE imports — eval/render scripts and the
real-robot pick loop all import this module.
"""

import json
import os
import shutil

from mujoco_playground import registry

DEFAULT_ENTITY = "weissma6-zhaw-school-of-engineering"
DEFAULT_PROJECT = "UR3_pick_ppo"

_CACHE_ROOT = os.path.dirname(os.path.abspath(__file__))


def default_policy_dir(run_id: str) -> str:
  """Resolve the run-id-keyed cache dir (cwd-independent, from this file)."""
  return os.path.join(_CACHE_ROOT, run_id)


def policy_exists_locally(run_id: str, out_dir: str = None) -> bool:
  """True iff both params.msgpack and metadata.json are cached for run_id."""
  out_dir = out_dir or default_policy_dir(run_id)
  return os.path.exists(os.path.join(out_dir, "params.msgpack")) and os.path.exists(
      os.path.join(out_dir, "metadata.json")
  )


def _fetch_from_wandb(run, out_dir):
  """Download params.msgpack + inference_config.json from a W&B run.

  Prefers the Files-tab `trained_policy/` folder, falls back to the run's
  `model` artifact. Returns (params_src_path, inference_cfg_path_or_None).
  """
  stage = os.path.join(out_dir, "_wandb")
  os.makedirs(stage, exist_ok=True)

  # Prefer the Files-tab trained_policy/ folder.
  files = {f.name: f for f in run.files()}
  params_name = "trained_policy/params.msgpack"
  inf_name = "trained_policy/inference_config.json"
  if params_name in files:
    files[params_name].download(root=stage, replace=True)
    params_src = os.path.join(stage, params_name)
    inf_cfg_path = None
    if inf_name in files:
      files[inf_name].download(root=stage, replace=True)
      inf_cfg_path = os.path.join(stage, inf_name)
    print("[wandb] fetched policy from Files (trained_policy/)")
    return params_src, inf_cfg_path

  # Fall back to the legacy `model` artifact.
  policy_art = next(
      (a for a in run.logged_artifacts() if a.type == "model"), None
  )
  if policy_art is None:
    raise ValueError(
        f"No 'trained_policy/' Files folder and no 'model' artifact found "
        f"for run {run.id}"
    )
  art_dir = policy_art.download(root=os.path.join(out_dir, "_artifact"))
  inf_cfg_path = os.path.join(art_dir, "inference_config.json")
  if not os.path.exists(inf_cfg_path):
    inf_cfg_path = None
  print("[wandb] fetched policy from the 'model' artifact (fallback)")
  return os.path.join(art_dir, "params.msgpack"), inf_cfg_path


def download_policy(
    run_id: str,
    out_dir: str = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    force: bool = False,
) -> str:
  """Download a trained PPO policy from W&B into a load_policy_fn-ready dir.

  Writes params.msgpack + a metadata.json (env_name + network_factory + obs/
  action dims from the run's inference_config.json, registry-resolved as a
  fallback). Skips the download if out_dir already has both files unless
  force=True. Returns out_dir.
  """
  out_dir = out_dir or default_policy_dir(run_id)
  if not force and policy_exists_locally(run_id, out_dir):
    print(f"[wandb] policy already present at {out_dir} (skipping download)")
    return out_dir

  import wandb

  os.makedirs(out_dir, exist_ok=True)
  api = wandb.Api()
  run = api.run(f"{entity}/{project}/{run_id}")

  params_src, inf_cfg_path = _fetch_from_wandb(run, out_dir)
  params_out = os.path.join(out_dir, "params.msgpack")
  shutil.copyfile(params_src, params_out)

  # Prefer the inference_config.json shipped with the policy (carries the real
  # network_factory + obs/action sizes); fall back to run.config + the registry
  # for older runs that predate self-describing policies.
  inf_cfg = {}
  if inf_cfg_path and os.path.exists(inf_cfg_path):
    with open(inf_cfg_path) as f:
      inf_cfg = json.load(f)

  env_name = inf_cfg.get("env_name") or run.config.get("env_name", "UR3PicknPlace")
  env = registry.load(env_name)
  reg_obs, reg_act = int(env.observation_size), int(env.action_size)

  if inf_cfg:
    nf = inf_cfg.get("network_factory", {}) or {}
    obs_dim = int(inf_cfg.get("observation_size", reg_obs))
    action_dim = int(inf_cfg.get("action_size", reg_act))
    if obs_dim != reg_obs or action_dim != reg_act:
      print(f"[wandb] WARNING: artifact obs/act ({obs_dim},{action_dim}) "
            f"!= env {env_name} ({reg_obs},{reg_act}); using artifact dims")
  else:
    nf = run.config.get("network_factory", {}) or {}
    obs_dim, action_dim = reg_obs, reg_act
    if not nf:
      print("[wandb] WARNING: no network_factory in policy or run.config; "
            "load may fail with a params shape mismatch")

  metadata = {
      "env_name": env_name,
      "obs_dim": obs_dim,
      "action_dim": action_dim,
      "network_factory": nf,
      "wandb_run_id": run_id,
      "wandb_entity": entity,
      "wandb_project": project,
      "action_scale": inf_cfg.get("action_scale",
                                  run.config.get("action_scale", 0.04)),
      "ctrl_dt": inf_cfg.get("ctrl_dt", run.config.get("ctrl_dt", 0.02)),
      "episode_length": inf_cfg.get("episode_length",
                                    run.config.get("episode_length")),
  }
  with open(os.path.join(out_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
  print(f"[wandb] downloaded policy -> {out_dir} "
        f"(obs={metadata['obs_dim']}, act={metadata['action_dim']})")
  return out_dir


if __name__ == "__main__":
  # Example: fetch one trained UR3 pick policy into its run-id cache dir.
  example_run_id = "Pick_tiny_5k_un10_env256_20260625_141404_6311"
  download_policy(example_run_id, project=DEFAULT_PROJECT)
