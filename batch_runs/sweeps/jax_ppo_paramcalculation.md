# JAX PPO Sweep Parameter Calculation (UR3Pick / UR3PicknPlace)

Reference for generating sweep JSONL lines in this folder. Runner:
`batch_runs/scripts/run_one_ur3.py` → `learning/notebooks/run_experiment.py` → `brax.training.agents.ppo.train`.

Every JSONL line is one run. Keys that are NOT in the reserved set below are passed
straight through as PPO overrides on top of the env defaults from
`mujoco_playground/config/manipulation_params.py` (`brax_ppo_config`).

---

## 1. The formula (verified against real runs)

```
env_step_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat
num_evals_after_init       = max(num_evals - 1, 1)
num_training_steps_per_epoch = ceil( num_timesteps /
                                     (num_evals_after_init * env_step_per_training_step * max(num_resets_per_eval, 1)) )

TOTAL_STEPS = num_evals_after_init * max(num_resets_per_eval,1)
              * num_training_steps_per_epoch * env_step_per_training_step
```

`num_timesteps` is only a **floor target** — `ceil()` means actual `TOTAL_STEPS` is
almost always **higher** than requested. Plan for the TOTAL, not the request.

Total gradient updates = `num_evals_after_init * max(num_resets_per_eval,1) * num_training_steps_per_epoch * num_updates_per_batch * num_minibatches`.

### ⚠️ The trap: `num_resets_per_eval`

UR3Pick's default is **`num_resets_per_eval = 10`** (inherited from the shared base
config, not overridden in the UR3Pick branch). It sits in the denominator AND as a
multiplier, so it inflates `TOTAL_STEPS` by 10× and forces a large minimum.

**Past tiny runs confirm it** (sweep had `num_envs=256`, defaults `batch_size=512,
num_minibatches=32, num_resets_per_eval=10`, `num_evals=1`):

| unroll | env_step/step | nts | TOTAL_STEPS (actual) |
|--------|---------------|-----|----------------------|
| 5  | 81 920  | 1 | 819 200   |
| 10 | 163 840 | 1 | 1 638 400 |
| 20 | 327 680 | 1 | 3 276 800 |

→ **To control total steps precisely (any target below ~20M), set
`"num_resets_per_eval": 1` in the JSONL.** Otherwise the 10× floor makes 8M impossible
(minimum would be ~22.9M for 15 evals).

---

## 2. UR3Pick defaults (the base you override)

From `manipulation_params.brax_ppo_config("UR3Pick")`:

```
num_timesteps        = 20_000_000      num_envs            = 2048
num_evals            = 4               batch_size          = 512
unroll_length        = 10              num_minibatches     = 32
num_updates_per_batch= 8               num_resets_per_eval = 10   ← from base config
discounting          = 0.97            learning_rate       = 1e-3
entropy_cost         = 2e-2            reward_scaling      = 1.0
episode_length       = 250             action_repeat       = 1
policy_hidden_layer_sizes=(32,32,32,32)  value=(256,256,256,256,256)
```

Constraint enforced by `ppo.train`: **`batch_size * num_minibatches % num_envs == 0`**.
(512*32 % 256 = 0, 512*16 % 256 = 0, 512*8 % 256 = 0 — all fine with num_envs=256.)
`num_envs` does NOT appear in the step formula; it only affects sharding + this assert.

---

## 3. Videos: `video_every_evals`

`policy_params_fn` runs once per eval (eval_idx = 1..num_evals). A video is recorded when
`eval_idx % video_every_evals == 0`, plus always on the final eval.

```
video_every_evals = num_evals / num_videos      (choose num_evals divisible by num_videos)
```

Example: `num_evals=15`, want 3 videos → `video_every_evals = 5` → videos at evals 5, 10, 15.

For a clean count, keep `num_evals` an exact multiple of `num_videos`. If not divisible,
the always-on last-eval video adds one extra.

---

## 4. Recipe: pick params for a target

1. **Always set `num_resets_per_eval = 1`** (predictable counts).
2. Keep `num_evals ≥ 15` for a smooth W&B curve.
3. `video_every_evals = num_evals / num_videos`.
4. Choose `env_step_per_training_step` so `num_timesteps / (num_evals_after_init * env_step)`
   lands near a whole number. The easiest knob is **`num_minibatches`** (smaller → finer steps).
   Keep `batch_size=512`, `num_envs=256`, `unroll_length=10` unless there's a reason.
5. Verify TOTAL_STEPS with the formula and report it (it will be ≥ target).

### Verified worked numbers (num_resets_per_eval=1, num_evals=15, bs=512, unroll=10)

| Target | num_minibatches | env_step/step | nts | TOTAL_STEPS | vs target |
|--------|-----------------|---------------|-----|-------------|-----------|
| 8M  | 32 | 163 840 | 4  | 9 175 040  | 115% |
| **8M**  | **16** | **81 920**  | **7**  | **8 028 160**  | **100% ✓ best** |
| 8M  | 8  | 40 960  | 14 | 8 028 160  | 100% ✓ |
| 15M | 32 | 163 840 | 7  | 16 056 320 | 107% |
| **15M** | **16** | **81 920**  | **14** | **16 056 320** | **107% ✓ best** |

---

## 5. Reserved keys (metadata, NOT passed to ppo.train)

`run_id, wandb_project, wandb_mode, wandb_group, wandb_tags, out_dir, video_every_evals,
render_every, video_tag, camera_kwargs, deterministic, env_name, algo, notes,
init_keyframe, num_eval_envs, seed, domain_randomization`

Everything else (num_timesteps, num_evals, unroll_length, num_envs, batch_size,
num_minibatches, num_updates_per_batch, num_resets_per_eval, learning_rate, entropy_cost,
discounting, reward_scaling, network_factory, …) is a PPO override.

---

## 6. Canonical JSONL template

One line per run. UR3Pick standard fields:

```json
{"run_id":"Pick_8M_15ev_3vid","env_name":"UR3Pick","init_keyframe":"low_home","camera_kwargs":{"camera":"box_detail"},"num_timesteps":8000000,"num_evals":15,"num_resets_per_eval":1,"unroll_length":10,"num_envs":256,"batch_size":512,"num_minibatches":16,"num_updates_per_batch":8,"video_every_evals":5,"render_every":1,"learning_rate":0.001,"entropy_cost":0.02,"discounting":0.97,"reward_scaling":1.0,"seed":0,"wandb_project":"UR3_pick_ppo","wandb_tags":["8M","15ev","3vid"]}
```

That line = ~8.03M steps, 15 evals, 3 videos (evals 5/10/15). Append to the right
`*_sweep.jsonl` and bump the SLURM `--array` to match the line count (see repo CLAUDE.md).
