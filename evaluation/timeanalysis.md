# Time analysis — UR3_pick_ppo (last 4 finished runs)

All times in seconds. `slurm_time` is hand-entered (SLURMTIME); every other
value is from W&B (`run.summary` / `run.config`). `setup_overhead` = W&B runtime − training − eval; `slurm_overhead` = SLURM time − W&B runtime.

| metric | Pick_tiny_5k_un10_env256_20260625_130451_6311 | Pick_tiny_10k_un10_env256_20260625_125915_6311 | pickOrient4cm_lr_low_longunroll_20260607_125238_2201 | ur3pick_DR_MFR_size_20260603_092056_3714 |
| --- | --- | --- | --- | --- |
| state | finished | finished | finished | finished |
| created_at | 2026-06-25T11:04:53Z | 2026-06-25T10:59:17Z | 2026-06-07T10:52:40Z | 2026-06-03T09:20:57Z |
| num_envs | 256 | 256 | 2048 | 2048 |
| unroll_length | 10 | 10 | 50 | 10 |
| num_timesteps_requested | 5000 | 10000 | 20000000 | 20000000 |
| num_steps_actual | 1638400 | 1638400 | 90112000 | 22937600 |
| wandb_step | 1638400 | 1638400 | 90112000 | 22937600 |
| slurm_time | 7 | 17 | 8 | 15 |
| wandb_runtime | 490.64 | 410.12 | 2911.65 | 971.15 |
| training_walltime | 221.45 | 190.92 | 2257.13 | 646.19 |
| eval_walltime | 97.36 | 73.92 | 84.10 | 96.50 |
| setup_overhead | 171.83 | 145.27 | 570.42 | 228.46 |
| slurm_overhead | -483.64 | -393.12 | -2903.65 | -956.15 |
| training_sps | 151227.25 | 154901.95 | 425959.27 | 417218.97 |
| eval_sps | 197.21 | 259.73 | 11814.46 | 11437.47 |

## Takeaways
- **Pick_tiny_5k_un10_env256_20260625_130451_6311** (1638400 env steps, 256 envs): W&B runtime 490.6 s (training 221.5 s / eval 97.4 s / setup 171.8 s) ; SLURM 7.0 s → overhead -483.6 s (-99% beyond W&B) ; training 151227 sps
- **Pick_tiny_10k_un10_env256_20260625_125915_6311** (1638400 env steps, 256 envs): W&B runtime 410.1 s (training 190.9 s / eval 73.9 s / setup 145.3 s) ; SLURM 17.0 s → overhead -393.1 s (-96% beyond W&B) ; training 154902 sps
- **pickOrient4cm_lr_low_longunroll_20260607_125238_2201** (90112000 env steps, 2048 envs): W&B runtime 2911.7 s (training 2257.1 s / eval 84.1 s / setup 570.4 s) ; SLURM 8.0 s → overhead -2903.7 s (-100% beyond W&B) ; training 425959 sps
- **ur3pick_DR_MFR_size_20260603_092056_3714** (22937600 env steps, 2048 envs): W&B runtime 971.2 s (training 646.2 s / eval 96.5 s / setup 228.5 s) ; SLURM 15.0 s → overhead -956.2 s (-98% beyond W&B) ; training 417219 sps
