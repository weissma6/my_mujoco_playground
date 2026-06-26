# Time analysis — UR3_pick_ppo (last 4 finished runs)

All times in seconds. `slurm_time` is hand-entered (SLURMTIME); every other
value is from W&B (`run.summary` / `run.config`). `setup_overhead` = W&B runtime − training − eval; `slurm_overhead` = SLURM time − W&B runtime.

| metric | Pick_10M_env4096_20260625_172835_6311 | Pick_10M_env2048_20260625_172637_6311 | Pick_10M_env1024_20260625_171431_6311 | Pick_10M_env512_20260625_171420_6311 |
| --- | --- | --- | --- | --- |
| state | finished | finished | finished | finished |
| created_at | 2026-06-25T15:28:37Z | 2026-06-25T15:26:39Z | 2026-06-25T15:14:33Z | 2026-06-25T15:14:22Z |
| num_envs | 4096 | 2048 | 1024 | 512 |
| unroll_length | 10 | 10 | 10 | 10 |
| num_timesteps_requested | 10000000 | 10000000 | 10000000 | 10000000 |
| num_steps_actual | 10321920 | 10321920 | 10321920 | 10321920 |
| wandb_step | 10321920 | 10321920 | 10321920 | 10321920 |
| slurm_time | 420 | 420 | 480 | 600 |
| wandb_runtime | 582.34 | 621.97 | 661.76 | 787.94 |
| training_walltime | 311.86 | 351.12 | 380.58 | 506.42 |
| eval_walltime | 96.50 | 96.40 | 103.45 | 106.76 |
| setup_overhead | 173.99 | 174.46 | 177.72 | 174.76 |
| slurm_overhead | -162.34 | -201.97 | -181.76 | -187.94 |
| training_sps | 45938.63 | 41515.93 | 34906.75 | 26240.35 |
| eval_sps | 11141.09 | 11634.69 | 11708.77 | 11732.75 |

## Takeaways
- **Pick_10M_env4096_20260625_172835_6311** (10321920 env steps, 4096 envs): W&B runtime 582.3 s (training 311.9 s / eval 96.5 s / setup 174.0 s) ; SLURM 420.0 s → overhead -162.3 s (-28% beyond W&B) ; training 45939 sps
- **Pick_10M_env2048_20260625_172637_6311** (10321920 env steps, 2048 envs): W&B runtime 622.0 s (training 351.1 s / eval 96.4 s / setup 174.5 s) ; SLURM 420.0 s → overhead -202.0 s (-32% beyond W&B) ; training 41516 sps
- **Pick_10M_env1024_20260625_171431_6311** (10321920 env steps, 1024 envs): W&B runtime 661.8 s (training 380.6 s / eval 103.5 s / setup 177.7 s) ; SLURM 480.0 s → overhead -181.8 s (-27% beyond W&B) ; training 34907 sps
- **Pick_10M_env512_20260625_171420_6311** (10321920 env steps, 512 envs): W&B runtime 787.9 s (training 506.4 s / eval 106.8 s / setup 174.8 s) ; SLURM 600.0 s → overhead -187.9 s (-24% beyond W&B) ; training 26240 sps
