# VT2 Results — reproducible tables and figures

Everything the thesis Results chapter reports is generated from this
directory. No number in the text is computed by hand, in a notebook, or in a
chat session.

## Regenerate everything

Run from the repository root, in order. All three steps are plain
NumPy/pandas/MuJoCo and are safe to run locally — no MJX, no GPU, no robot.

```bash
# 1. Merge the raw run folders into the tidy long-format table.
#    Applies the campaign filter (see "Campaign filter" below).
python evaluation/gap_metrics.py --build          # -> evaluation/gap_metrics.csv

# 2. Dynamics replay over every completed real trial (plain MuJoCo mj_step).
python evaluation/run_f6_replay.py                # -> evaluation/f6_replay_summary.csv

# 3. Tables, figures, provenance.
python evaluation/report/build_report.py          # -> tables/ figures/
python evaluation/report/build_report.py --tables # tables only (fast, no replotting)
```

Step 3 also verifies its own inputs and **exits rather than emitting numbers**
if the table is not campaign-clean.

## What lands where

| Path | Contents |
|---|---|
| `tables/tN_*.tex` | Paste-ready LaTeX, ZHAW template style. `\input{Tables/tN_*}` |
| `tables/tN_*.csv` | The same numbers, machine-readable |
| `tables/t4_gripper_geometry.json` | Jaw opening and yaw admissibility, parsed from the model XML |
| `figures/*.pdf`, `*.png` | Report figures, hyphenated lowercase names |
| `report_numbers.json` | Every scalar quoted in the prose, keyed by name |
| `PROVENANCE.txt` | git commit, input SHA-256, row counts |

Copy `figures/*.pdf` into the Overleaf `Figures/` folder (the thesis vault
mirrors them in `Figures/Figures_final/`).

## Table index

| File | Label | Reports |
|---|---|---|
| `t1_dataset` | `tab:dataset` | Trials per configuration, cube and domain |
| `t2_retention` | `tab:retention` | Return retention, CI, noise floor, reportability gate |
| `t3_stage_retention` | `tab:stage_retention` | Retention grouped by task stage |
| `t4_abort` | `tab:abort` | Aborted trials + the gripper-clearance explanation |
| `t5_replay` | `tab:replay` | One-step and open-loop dynamics-replay fidelity |
| `t6_place_error` | `tab:place_error` | Placement accuracy (recomputed, see below) |
| `t7_regret` | `tab:regret` | Sim-selection regret |
| `t8_sim_success` | `tab:sim_success` | Task success in simulation |

## Campaign filter — why it exists

`robots/UR3e/real_robot_results/gap_protocol/` holds **two** campaigns:

| | 2026-07-22 (superseded) | 2026-07-29 (D23) |
|---|---|---|
| policies | `L*_s?_20260717_*`, 26-D obs | `L*_vel_s1_20260729_*`, 33-D obs |
| table | stale geometry | real 95 mm table |
| git | `1c624db3` | `15426ef3` |
| leaf names | `ep0_rep1_3cm` | `A1_ep0_rep1_3cm` |

Mixing them is not merely noisy, it **corrupts run totals**. Both campaigns
reuse `episode_id` 0–2 *and* seed 1, so a legacy and a D23 trial share an
identical `(config_id, domain, seed, episode_id, repeat, cube_size)` key —
and `_run_totals` groups on exactly that key and sums. An unfiltered build
merges 290 term-rows into fabricated runs with roughly double the return.

Three rules therefore apply, all in `gap_metrics.build_long_csv`:

1. **Policy match** — a trial is kept only if its `policy_run_id` matches
   `robots/UR3e/gap_protocol_policy_map.json`. A sim/real pair is only
   meaningful if both sides ran the same checkpoint.
2. **No cross-campaign mirroring** — 12 sim rollouts use a D23 policy but were
   initialized from a *legacy* trial's `measured_init.json`, pairing sim
   policy A against robot policy B. Dropped.
3. **Superseded re-runs** — `L0_none` episodes 3–5 were executed twice on
   2026-07-29, before and after the condition-prefix naming convention. The
   condition-prefixed execution is kept (canonical, later, and complete where
   the earlier batch aborted).

A hard assertion then fails the build if any run-identity key is still
duplicated, so this class of bug cannot recur silently.

After filtering: **135 real trials (5 × 27 = 9 conditions × 3 repeats), 91 of
them complete, 91 sim rollouts** — matching the campaign's own accounting.

## Two quantities recomputed here

**Place error (D22).** The value logged by `robots/UR3e/run_gap_protocol.py`
is **invalid**. It takes the landed cube pose straight from
`mocap.get_rigid_body_pose()` — mocap-*world* coordinates — and scores it
against `DROP_SQUARE_CENTER`, a robot *base*-frame point. Every logged value
therefore carries a fixed offset of roughly 0.3 m and every trial reads as
outside the square. This is the same missed-calibration class as D25.
`build_report.t_place_error` recovers it by applying the base-frame
calibration to the stored landing position; world *z* is taken from the
trial's own measured initial cube height, and since the calibration rotation
is within about a degree of a 180° turn about *z*, a ±50 mm error in that
assumption moves the result by under 1 mm.

> **Fix at source before the next campaign:** `run_gap_protocol.py:705` should
> transform `landed_xyz` into the base frame before calling `place_error()`.

**Gripper clearance.** Parsed from
`mujoco_playground/_src/manipulation/my_ur3/universal_robots_ur3e/ur3e_position.xml`
rather than hardcoded, so it cannot drift from the simulated gripper. Jaw
opening is the face-to-face distance of the two pad geoms at zero closure; a
cube of side *w* at yaw θ spans `w(|cos θ| + |sin θ|)`, so `w·√2 ≤ opening`
means every orientation is graspable.

## Related scripts (outside this directory)

| Script | Role |
|---|---|
| `evaluation/gap_metrics.py` | Metric layer + `--build`. `--selftest` guards its invariants |
| `evaluation/plots_gap.py` | Figure functions (no metric math lives here) |
| `evaluation/run_gap_plots.py` | Renders the figures that need only `gap_metrics.csv` |
| `evaluation/run_f6_replay.py` | Batch dynamics replay -> `f6_replay_summary.csv` |
| `evaluation/ur3_reward_replay.py` | Reconstructs reward terms from logged geometry (plain FK) |
| `evaluation/ur3_dynamics_replay.py` | One-step / open-loop replay core |
| `evaluation/run_gap_protocol_sim.py` | Commit 7 sim mirror (MJX — **HPC only**) |

## Not generated here

- **F3** per-step exact-parity trace — needs per-step arrays, not episode sums.
- **F7** timing honesty — needs the gripper-thread-decoupling commit identified.
- **F8** leave-one-cluster-out attribution — needs sim configs never trained.
