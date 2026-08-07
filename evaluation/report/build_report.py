"""Single reproducible entry point for every NUMBER and FIGURE in the VT2
Results chapter.

HARD RULE this file exists to satisfy: every table, every figure, and every
number quoted in the thesis prose must be retraceable to committed code in
this repository. Nothing is computed by hand, in a notebook, or in a chat.

    python evaluation/report/build_report.py            # tables + figures
    python evaluation/report/build_report.py --tables   # tables only (fast)

Outputs (all under evaluation/report/, all overwritten on each run):

    tables/*.tex        LaTeX, ZHAW-template style (plain \\hline, \\texttt{})
                        -> paste into Chapters/4_Results.tex
    tables/*.csv        the same numbers, machine-readable
    figures/*.pdf|png   report figures, hyphenated lowercase names matching
                        the Figures_final/ convention
    report_numbers.json every scalar quoted in the prose, keyed by the name
                        used in the text, so a reader can grep for it
    PROVENANCE.txt      git SHA, input file SHA-256, row counts, timestamps

Input contract
--------------
Reads evaluation/gap_metrics.csv, which is built by

    python evaluation/gap_metrics.py --build

from the raw run folders. That build applies the campaign filter (see the
CAMPAIGN FILTER note in gap_metrics.py): the results tree holds BOTH the
superseded 2026-07-22 campaign and the D23 2026-07-29 campaign, and mixing
them silently corrupts run totals. This script asserts the loaded table is
D23-only and refuses to emit anything if it is not.

Two derived quantities are recomputed HERE rather than read from the logs,
both documented at their call sites below:
  * place_error (D22) -- the logged value is invalid (frame bug, see
    _place_error_table), recomputed from landed_box_xy + the base-frame
    calibration.
  * gripper clearance / yaw admissibility -- parsed from the model XML, never
    hardcoded, so it cannot drift from the simulated gripper.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(_THIS_DIR)
REPO_ROOT = os.path.dirname(EVAL_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation import gap_metrics as gm  # noqa: E402

CSV_PATH = os.path.join(EVAL_DIR, "gap_metrics.csv")
F6_CSV_PATH = os.path.join(EVAL_DIR, "f6_replay_summary.csv")
REAL_ROOT = os.path.join(REPO_ROOT, "robots", "UR3e", "real_robot_results",
                          "gap_protocol")
XML_PATH = os.path.join(REPO_ROOT, "mujoco_playground", "_src", "manipulation",
                        "my_ur3", "universal_robots_ur3e", "ur3e_position.xml")
TABLES_DIR = os.path.join(_THIS_DIR, "tables")
FIGURES_DIR = os.path.join(_THIS_DIR, "figures")

LEVELS = ["L0_none", "L1_pos", "L2_pos_cube", "L3_pos_cube_robot", "L4_full"]
# Display names used in every table/figure caption in the thesis.
PRETTY = {
    "L0_none": "L0 (none)", "L1_pos": "L1 (pos)", "L2_pos_cube": "L2 (pos+cube)",
    "L3_pos_cube_robot": "L3 (pos+cube+robot)", "L4_full": "L4 (full)",
}
PRIMARY_CUBE = "3cm"
N_BOOT = 10000
BOOT_SEED = 0

NUMBERS = {}   # every scalar quoted in prose -> report_numbers.json


def _rec(key, value, note=""):
    """Record a number that the prose quotes, so it stays traceable."""
    NUMBERS[key] = {"value": value, "note": note}
    return value


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args, default="unknown"):
    try:
        return subprocess.check_output(["git", "-C", REPO_ROOT, *args],
                                        text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return default


def _fmt(v, nd=2, thousands=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if thousands:
        return f"{v:,.{nd}f}"
    return f"{v:.{nd}f}"


def tex_escape(s):
    """Escape LaTeX specials in a string that is inserted into text mode.

    Needed for anything interpolated from the filesystem or from a config
    key: a bare `_` in text mode is a subscript and aborts the compile with
    \"Missing $ inserted\". Only applied to interpolated VALUES, never to the
    hand-written LaTeX around them.
    """
    out = []
    for ch in str(s):
        if ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append("\\textasciitilde{}")
        elif ch == "^":
            out.append("\\textasciicircum{}")
        elif ch == "\\":
            out.append("\\textbackslash{}")
        else:
            out.append(ch)
    return "".join(out)


def _latex_table(caption, label, colspec, header, rows, note=None):
    """A float containing one tabular, optionally followed by a small note.

    The note is set inside a minipage rather than after a bare `\\\\`: a `\\\\`
    directly after `\\end{tabular}` inside a float raises \"There's no line
    here to end\" and kills the compile.
    """
    out = ["\\begin{table}[h]", "\\centering", f"\\caption{{{caption}}}",
           f"\\label{{{label}}}", f"\\begin{{tabular}}{{{colspec}}}", "\\hline",
           " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\hline"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\hline", "\\end{tabular}"]
    if note:
        out += ["", "\\vspace{2pt}", "\\begin{minipage}{0.95\\linewidth}",
                f"\\footnotesize {note}", "\\end{minipage}"]
    out += ["\\end{table}"]
    return "\n".join(out)


def lint_tex(name, tex):
    """Catch the LaTeX errors that produce NO PDF AT ALL, before they reach
    Overleaf. Returns a list of problem strings (empty = clean).

    All three of these have actually shipped from this script:
      * a bare `_` in text mode          -> \"Missing $ inserted\" (fatal)
      * `\\\\` right after `\\end{tabular}` -> \"There's no line here to end\" (fatal)
      * a row whose column count differs from the preamble -> misaligned or
        \"Extra alignment tab\" (fatal)
    """
    import re as _re
    problems = []
    lines = tex.split("\n")

    # Arguments of these commands are never typeset, so an underscore in them
    # is harmless -- the Method chapter already ships \label{tab:dr_clusters}.
    _ref_cmd = _re.compile(
        r"\\(label|ref|eqref|cite|citep|citet|input|include|includegraphics"
        r"|autoref|nameref)(\[[^\]]*\])?\{[^}]*\}")

    for i, ln in enumerate(lines, 1):
        if ln.strip().startswith("%"):
            continue
        ln = _ref_cmd.sub("", ln)
        # strip math spans, then look for unescaped _ ^ &
        stripped, in_math, prev = [], False, ""
        for ch in ln:
            if ch == "$" and prev != "\\":
                in_math = not in_math
            elif not in_math:
                stripped.append((ch, prev))
            prev = ch
        for ch, pv in stripped:
            if ch == "_" and pv != "\\":
                problems.append(f"{name}:{i}: unescaped '_' in text mode")
                break
        if ln.count("$") % 2:
            problems.append(f"{name}:{i}: odd number of '$' on the line")

    for i, ln in enumerate(lines):
        if ln.strip() == "\\end{tabular}":
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt.startswith("\\\\"):
                problems.append(f"{name}:{i+2}: '\\\\' directly after "
                                 f"\\end{{tabular}} is a fatal error")

    # column count: preamble letters vs '&' per body row
    for i, ln in enumerate(lines):
        if ln.startswith("\\begin{tabular}"):
            spec = ln[ln.index("{", len("\\begin{tabular}")) + 1: ln.rindex("}")]
            # Fixed-width cells (p{2.6cm}, m{...}, b{...}) are ONE column each,
            # but their width text can itself contain 'c' (as in "2.6cm") or
            # 'r'/'l' -- count and strip them BEFORE counting bare l/r/c
            # letters, or a column spec like "p{2.6cm}p{3.0cm}" is miscounted
            # (the two 'c's in "cm"x2 inflated a real 4-column table to 8).
            fixed_width = _re.findall(r"[pmb]\{[^}]*\}", spec)
            remainder = _re.sub(r"[pmb]\{[^}]*\}", "", spec)
            remainder = _re.sub(r"@\{[^}]*\}", "", remainder)  # column glue, not a column
            ncol = len(fixed_width) + sum(1 for c in remainder if c in "lrc")
            for j in range(i + 1, len(lines)):
                b = lines[j]
                if b.startswith("\\end{tabular}"):
                    break
                if "&" not in b or not b.rstrip().endswith("\\\\"):
                    continue
                got = b.count("&") - b.count("\\&") + 1
                if got != ncol:
                    problems.append(f"{name}:{j+1}: row has {got} columns, "
                                     f"preamble declares {ncol}")
    if tex.count("{") != tex.count("}"):
        problems.append(f"{name}: unbalanced braces "
                         f"({tex.count('{')} open, {tex.count('}')} close)")
    return problems


_ALL_TEX = []   # every table, in emission order, for the single-file bundle
LINT_PROBLEMS = []


def _write(name, tex, df):
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, name + ".tex"), "w", encoding="utf-8") as f:
        f.write(tex + "\n")
    df.to_csv(os.path.join(TABLES_DIR, name + ".csv"), index=False)
    _ALL_TEX.append((name, tex))
    problems = lint_tex(name, tex)
    LINT_PROBLEMS.extend(problems)
    flag = "" if not problems else f"  [{len(problems)} LATEX PROBLEM(S)]"
    print(f"  wrote tables/{name}.tex + .csv{flag}")
    for p in problems:
        print(f"      !! {p}")


def _write_bundle():
    """One self-contained file with every table inlined.

    Overleaf projects that do not carry a Tables/ subdirectory cannot use
    \\input{}: a missing \\input target is a FATAL LaTeX error and produces no
    PDF at all. Pasting this bundle (or the individual tables inline) removes
    that failure mode entirely.
    """
    order = ["t1_dataset", "t2_retention", "t3_stage_retention", "t4_abort",
             "t5_replay", "t6_place_error", "t7_regret", "t8_sim_success",
             "t9_coverage", "t10_condition_split"]
    by_name = dict(_ALL_TEX)
    parts = ["% " + "=" * 68,
             "% VT2 Results -- all tables, generated by",
             "% evaluation/report/build_report.py. Do not edit by hand;",
             "% re-run the script instead.",
             "% " + "=" * 68, ""]
    for n in order:
        if n in by_name:
            parts += [f"% ---- {n} " + "-" * (60 - len(n)), by_name[n], ""]
    with open(os.path.join(TABLES_DIR, "all_tables.tex"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"  wrote tables/all_tables.tex ({len(by_name)} tables inlined)")
    if LINT_PROBLEMS:
        raise SystemExit(
            f"\n{len(LINT_PROBLEMS)} LaTeX problem(s) would break the "
            f"Overleaf compile; refusing to emit. Fix them above.")
    print("  LaTeX lint: all tables clean")


# ===========================================================================
# Guard: the loaded table must be the D23 campaign only.
# ===========================================================================


def _assert_d23_only(df):
    pm = gm.load_policy_map()
    if not pm:
        raise SystemExit("no gap_protocol_policy_map.json -- cannot verify the "
                          "campaign filter; refusing to emit report numbers.")
    if "policy_run_id" not in df.columns:
        raise SystemExit(
            f"{CSV_PATH} has no policy_run_id column -- it predates the campaign "
            f"filter. Rebuild with: python evaluation/gap_metrics.py --build")
    bad = sorted(set(df["policy_run_id"].dropna()) - set(pm.values()))
    if bad:
        raise SystemExit(f"non-D23 policies present in {CSV_PATH}: {bad}. "
                          f"Rebuild with: python evaluation/gap_metrics.py --build")
    kcols = ["config_id", "domain", "seed", "episode_id", "repeat", "cube_size",
             "term"]
    n_dup = int((df.groupby(kcols, dropna=False).size() > 1).sum())
    if n_dup:
        raise SystemExit(f"{n_dup} duplicated run-identity keys -- run totals "
                          f"would be fabricated. Rebuild the CSV.")
    print(f"  campaign check OK: D23 only, no run-identity collisions")


# ===========================================================================
# T1 -- dataset summary
# ===========================================================================


def t_dataset(df):
    runs = df.drop_duplicates(["config_id", "domain", "episode_id", "repeat",
                                "cube_size"])
    rows, recs = [], []
    for c in LEVELS:
        r3 = runs[(runs.config_id == c) & (runs.domain == "real") &
                  (runs.cube_size == "3cm")]
        r4 = runs[(runs.config_id == c) & (runs.domain == "real") &
                  (runs.cube_size == "4cm")]
        s = runs[(runs.config_id == c) & (runs.domain == "sim")]
        rec = dict(config=c, real_3cm=len(r3),
                   real_3cm_complete=int((r3.stop_reason == "complete").sum()),
                   real_4cm=len(r4),
                   real_4cm_complete=int((r4.stop_reason == "complete").sum()),
                   sim=len(s))
        recs.append(rec)
        rows.append([PRETTY[c], str(rec["real_3cm"]), str(rec["real_3cm_complete"]),
                     str(rec["real_4cm"]), str(rec["real_4cm_complete"]),
                     str(rec["sim"])])
    d = pd.DataFrame(recs)
    tot = ["\\textbf{Total}"] + [f"\\textbf{{{d[k].sum()}}}" for k in
                                 ["real_3cm", "real_3cm_complete", "real_4cm",
                                  "real_4cm_complete", "sim"]]
    rows.append(tot)
    _rec("n_real_runs", int(d.real_3cm.sum() + d.real_4cm.sum()),
         "real robot trials in the D23 campaign")
    _rec("n_real_complete", int(d.real_3cm_complete.sum() + d.real_4cm_complete.sum()),
         "real trials completing all H=400 steps")
    _rec("n_sim_runs", int(d.sim.sum()), "matched-initialisation sim rollouts")
    tex = _latex_table(
        "Trials in the D23 evaluation campaign, after the campaign filter. "
        "\\enquote{Compl.} counts trials that ran all $H=400$ steps; the "
        "remainder aborted (protective stop, RTDE error, mocap loss or hang "
        "watchdog). Sim rollouts mirror only completed real trials.",
        "tab:dataset",
        "lrrrrr", ["Configuration", "Real 3\\,cm", "Compl.", "Real 4\\,cm",
                   "Compl.", "Sim"], rows)
    _write("t1_dataset", tex, d)
    return d


# ===========================================================================
# T2 -- return retention + noise floor gate
# ===========================================================================


def t_retention(df, cube=PRIMARY_CUBE):
    # 2026-07-30: headline retention/floor MUST be scored on the exact-parity
    # term subset (D8) -- 3 Method.md's exact-parity paragraph promises this
    # explicitly, and gm.retention()/gm.noise_floor() only do so if `terms=`
    # is passed (found: EXACT_PARITY_TERMS was defined but never applied
    # anywhere upstream, so this table was silently reporting full-reward
    # retention until this fix).
    rows, recs = [], []
    for c in LEVELS:
        r = gm.retention(df, c, cube_size=cube, n_boot=N_BOOT, seed=BOOT_SEED,
                          terms=gm.EXACT_PARITY_TERMS)
        nf = gm.noise_floor(df, c, cube_size=cube,
                             terms=gm.EXACT_PARITY_TERMS)["noise_floor"]
        piv = gm.per_episode_returns(df, c, cube_size=cube,
                                      terms=gm.EXACT_PARITY_TERMS)
        if len(piv):
            sim, real = piv["sim"].to_numpy(), piv["real"].to_numpy()
            rng = np.random.default_rng(BOOT_SEED)
            idx = gm._boot_episode_indices(len(piv), N_BOOT, rng)
            gaps = np.mean(sim[idx] - real[idx], axis=1)
            ci = gm._percentile_ci(gaps)
            gap = float(np.mean(sim - real))
            verdict = gm.gate_against_noise_floor(gap, ci, nf)
        else:
            gap, ci, verdict = np.nan, (np.nan, np.nan), "no paired data"
        recs.append(dict(config=c, n_episodes=r["n_episodes"], R_sim=r["R_sim"],
                         R_real=r["R_real"], retention=r["retention"],
                         ret_ci_lo=r["ci95"][0], ret_ci_hi=r["ci95"][1],
                         gap=gap, gap_ci_lo=ci[0], gap_ci_hi=ci[1],
                         noise_floor=nf, gate=verdict))
        rows.append([
            PRETTY[c], str(r["n_episodes"]), _fmt(r["R_sim"], 0),
            _fmt(r["R_real"], 0), _fmt(r["retention"], 2, False),
            f"[{_fmt(r['ci95'][0],2,False)}, {_fmt(r['ci95'][1],2,False)}]",
            _fmt(gap, 0), _fmt(nf, 0),
            "yes" if verdict == "reportable" else "no",
        ])
        _rec(f"retention_{c}", None if not np.isfinite(r["retention"]) else
             round(float(r["retention"]), 3), f"real/sim return retention, {cube}")
        _rec(f"gate_{c}", verdict, f"noise-floor gate verdict, {cube}")
    d = pd.DataFrame(recs)
    tex = _latex_table(
        f"Return retention on the {cube.replace('cm', '~cm')} "
        "(in-distribution) cube. $R$ is the mean episode return, summed over "
        "the exact-parity reward subset only (\\texttt{gripper\\_box}, "
        "\\texttt{approach\\_open}, \\texttt{gripper\\_align}, \\texttt{grasp}, "
        "\\texttt{lift}, \\texttt{action\\_rate}), over paired episodes; "
        "retention is $R_{\\mathrm{real}}/R_{\\mathrm{sim}}$ on raw returns "
        "with a paired bootstrap resampling episodes. "
        "\\enquote{Above floor} is the D9 reportability gate: yes only if the "
        "95\\,\\% interval of the gap lies entirely outside the "
        "$\\pm$noise-floor band.",
        "tab:retention", "lrrrrcrrc",
        ["Configuration", "$n_{\\mathrm{ep}}$", "$R_{\\mathrm{sim}}$",
         "$R_{\\mathrm{real}}$", "Retention", "95\\,\\% CI", "Gap",
         "Floor", "Above floor"], rows)
    _write("t2_retention", tex, d)
    return d


# ===========================================================================
# T3 -- per-stage retention
# ===========================================================================

STAGE_LABEL = {"approach": "Approach", "grasp": "Grasp", "lift": "Lift",
               "transport": "Transport", "regularizer": "Regularizer"}


def t_stage_retention(df, cube=PRIMARY_CUBE):
    per = {c: gm.term_retention(df, c, cube_size=cube)["per_stage"]["retention"]
           for c in LEVELS}
    d = pd.DataFrame(per)
    order = [s for s in gm.STAGE_ORDER if s in d.index]
    # regularizer last reads better: it is the only stage above 1.0
    order = [s for s in order if s != "regularizer"] + \
            (["regularizer"] if "regularizer" in d.index else [])
    rows = []
    for s in order:
        rows.append([STAGE_LABEL.get(s, s)] +
                    [_fmt(d.loc[s, c], 2, False) if np.isfinite(d.loc[s, c])
                     else "--" for c in LEVELS])
        for c in LEVELS:
            v = d.loc[s, c]
            _rec(f"stage_retention_{s}_{c}",
                 None if not np.isfinite(v) else round(float(v), 3),
                 f"per-stage retention, {cube}")
    tex = _latex_table(
        f"Per-stage return retention $R_{{\\mathrm{{real}}}}/R_{{\\mathrm{{sim}}}}$ "
        f"on the {cube.replace('cm','~cm')} cube, reward terms grouped by task "
        "stage. A dash marks a stage whose simulated return is identically "
        "zero, so the ratio is undefined: $L0$ never grasps in either domain.",
        "tab:stage_retention", "l" + "r" * len(LEVELS),
        ["Stage"] + [PRETTY[c] for c in LEVELS], rows,
        note="Terms per stage: approach = \\texttt{gripper\\_box}, "
             "\\texttt{approach\\_open}, \\texttt{gripper\\_align}; grasp = "
             "\\texttt{grasp}; lift = \\texttt{lift}; transport = "
             "\\texttt{box\\_target}, \\texttt{hold\\_target}; regularizer = "
             "\\texttt{no\\_floor\\_collision}, \\texttt{robot\\_target\\_qpos}, "
             "\\texttt{action\\_rate}.")
    out = d.reset_index().rename(columns={"index": "stage"})
    _write("t3_stage_retention", tex, out)
    return d


# ===========================================================================
# T4 -- abort rate + gripper admissibility (the mechanical explanation)
# ===========================================================================


def _gripper_geometry():
    """Pad-to-pad opening and yaw admissibility, PARSED from the model XML so
    it can never drift from the simulated gripper.

    The Hand-E fingers are two opposed slide joints. Each pad is a box geom;
    the inner face of a pad is its centre offset minus/plus its half-extent
    along the travel axis, and the reachable opening is the face-to-face
    distance at zero closure.

    A cube of side w presented at yaw theta about the approach axis spans
    w*(|cos t| + |sin t|) across the jaws, so it is admissible only where
    that stays below the opening. w*sqrt(2) <= opening means every yaw is
    admissible.
    """
    root = ET.parse(XML_PATH).getroot()
    geoms = {g.get("name"): g for g in root.iter("geom")}
    lf, rf = geoms["left_finger_collision"], geoms["right_finger_collision"]

    def _xhalf(g):
        return float(g.get("size").split()[0])

    def _xpos(g):
        return float(g.get("pos").split()[0])

    inner_left = _xpos(lf) + _xhalf(lf)     # pad face pointing +x
    inner_right = _xpos(rf) - _xhalf(rf)    # pad face pointing -x
    opening = inner_right - inner_left
    out = {"opening_m": opening, "xml": os.path.relpath(XML_PATH, REPO_ROOT)}
    for w, key in ((0.030, "3cm"), (0.040, "4cm")):
        diag = w * math.sqrt(2)
        per_side = (opening - w) / 2.0
        if diag <= opening:
            adm_deg, frac = 90.0, 1.0
        else:
            adm_deg = math.degrees(
                math.asin(min(1.0, (opening / w) / math.sqrt(2))) - math.pi / 4)
            frac = 2 * adm_deg / 90.0
        out[key] = {"side_m": w, "diagonal_m": diag, "clearance_per_side_m": per_side,
                    "admissible_halfangle_deg": adm_deg, "admissible_fraction": frac}
    return out


def t_abort(df):
    geo = _gripper_geometry()
    _rec("gripper_opening_mm", round(geo["opening_m"] * 1000, 1),
         "Hand-E pad-to-pad opening, parsed from the model XML")
    for k in ("3cm", "4cm"):
        _rec(f"clearance_per_side_mm_{k}",
             round(geo[k]["clearance_per_side_m"] * 1000, 2),
             "per-side clearance at aligned yaw")
        _rec(f"admissible_yaw_deg_{k}", round(geo[k]["admissible_halfangle_deg"], 1),
             "half-angle of graspable yaw about each 90 deg symmetry")
        _rec(f"admissible_yaw_fraction_{k}", round(geo[k]["admissible_fraction"], 3),
             "fraction of yaw angles the jaws can close over")

    rows, recs = [], []
    for c in LEVELS:
        cells = []
        for cube in ("3cm", "4cm"):
            a = gm.abort_rate(df, c, cube_size=cube)
            recs.append(dict(config=c, cube=cube, n_runs=a["n_runs"],
                             n_aborted=a["n_aborted"], abort_rate=a["abort_rate"]))
            cells.append(f"{a['n_aborted']}/{a['n_runs']}")
            _rec(f"abort_{c}_{cube}", f"{a['n_aborted']}/{a['n_runs']}",
                 "aborted / attempted real trials")
        rows.append([PRETTY[c]] + cells)
    d = pd.DataFrame(recs)
    g3, g4 = geo["3cm"], geo["4cm"]
    tex = _latex_table(
        "Aborted real trials per configuration and cube, as "
        "aborted/attempted. An abort is a protective stop, RTDE error, mocap "
        "loss or hang-watchdog trip; simulation never aborts. $L0$ never "
        "closes the gripper, so it cannot fail a grasp -- its 4\\,cm column is "
        "not evidence of robustness.",
        "tab:abort", "lcc", ["Configuration", "3\\,cm cube", "4\\,cm cube"], rows,
        note=(f"Jaw opening is {geo['opening_m']*1000:.1f}~mm "
              f"(\\texttt{{{tex_escape(geo['xml'])}}}). The 3\\,cm cube leaves "
              f"{g3['clearance_per_side_m']*1000:.1f}~mm per side and its "
              f"diagonal ({g3['diagonal_m']*1000:.1f}~mm) still fits, so every "
              f"yaw is graspable. The 4\\,cm cube leaves only "
              f"{g4['clearance_per_side_m']*1000:.1f}~mm per side and its "
              f"diagonal ({g4['diagonal_m']*1000:.1f}~mm) exceeds the opening, "
              f"so the jaws can only close over "
              f"$\\pm{g4['admissible_halfangle_deg']:.0f}^\\circ$ of yaw about "
              f"each $90^\\circ$ symmetry, i.e.\\ "
              f"{g4['admissible_fraction']*100:.0f}\\,\\% of orientations."))
    _write("t4_abort", tex, d)
    with open(os.path.join(TABLES_DIR, "t4_gripper_geometry.json"), "w") as f:
        json.dump(geo, f, indent=2)
    return d, geo


# ===========================================================================
# T5 -- dynamics replay (F6)
# ===========================================================================


def t_replay():
    if not os.path.exists(F6_CSV_PATH):
        print(f"  SKIP replay table: {F6_CSV_PATH} missing "
              f"(run evaluation/run_f6_replay.py)")
        return None
    s = pd.read_csv(F6_CSV_PATH)
    rows, recs = [], []
    for c in LEVELS:
        sub = s[s.config_id == c]
        if not len(sub):
            continue
        tcp_mm = 1000 * sub.onestep_tcp_rms_m.median()
        e1 = sub.E_replay_onestep_joint_rad2.median()
        ttd = sub.time_to_divergence_step.median()
        ttd_s = sub.time_to_divergence_s.median()
        n_never = int(sub.time_to_divergence_step.isna().sum())
        recs.append(dict(config=c, n=len(sub), tcp_rms_median_mm=tcp_mm,
                         E_onestep_median_rad2=e1, ttd_median_steps=ttd,
                         ttd_median_s=ttd_s, n_never_diverged=n_never))
        rows.append([PRETTY[c], str(len(sub)), _fmt(tcp_mm, 2, False),
                     f"{e1:.2e}",
                     "--" if not np.isfinite(ttd) else _fmt(ttd, 0),
                     "--" if not np.isfinite(ttd_s) else _fmt(ttd_s, 2, False),
                     str(n_never)])
        _rec(f"tcp_rms_mm_{c}", round(float(tcp_mm), 2),
             "median one-step replay TCP error")
        _rec(f"ttd_steps_{c}", None if not np.isfinite(ttd) else float(ttd),
             "median open-loop time-to-divergence")
    d = pd.DataFrame(recs)
    _rec("tcp_rms_mm_min", round(float(d.tcp_rms_median_mm.min()), 2), "")
    _rec("tcp_rms_mm_max", round(float(d.tcp_rms_median_mm.max()), 2), "")
    tex = _latex_table(
        "Dynamics-replay fidelity over completed real trials. One-step error "
        "re-synchronises the model to the measured state every control tick "
        "and integrates a single tick; open-loop sets the measured initial "
        "state once and then replays the whole logged command sequence "
        "without correction. Time-to-divergence is the first tick at which "
        "open-loop tool-centre-point error exceeds 1\\,cm.",
        "tab:replay", "lrrrrrr",
        ["Configuration", "$n$", "One-step TCP (mm)",
         "$E_{\\mathrm{replay}}$ (rad$^2$)", "TTD (steps)", "TTD (s)",
         "Never div."], rows,
        note="Replay is driven by the logged servoJ targets, so the policy is "
             "not in the feedback loop; it measures model-versus-robot error "
             "along the states that policy visited, and is therefore not a "
             "policy-independent quantity. Time-to-divergence is not "
             "comparable across configurations, since each policy generates a "
             "different trajectory.")
    _write("t5_replay", tex, d)
    return d


# ===========================================================================
# T6 -- place error (D22), RECOMPUTED (the logged value is invalid)
# ===========================================================================


def t_place_error(df):
    """The logged place_error_m / place_in_square are INVALID and must not be
    reported as-is.

    robots/UR3e/run_gap_protocol.py takes the landed cube pose straight from
    `mocap.get_rigid_body_pose()` -- mocap-WORLD coordinates -- and scores it
    against DROP_SQUARE_CENTER, which gap_metrics.py documents as a robot
    BASE-frame point. The two frames differ by the base-frame calibration, so
    every logged value carries a fixed ~0.3 m offset and every trial reads as
    outside the square. This is the same missed-calibration class as D25.

    Recovery: apply the calibration to the stored landed_box_xy. The stored
    value is 2-D, so the world z is taken from the trial's own measured
    initial cube height (the cube rests on the same surface before the pick
    and after the drop). The calibration rotation is within about one degree
    of a 180 degree turn about z, so the world-z contribution to base-frame
    x,y is ~0.01: a +/-50 mm error in the assumed height moves the result by
    under 1 mm, far below the 75 mm square half-width.
    """
    import glob
    from evaluation.run_gap_protocol_sim import (load_base_calibration,
                                                  mocap_pos_to_base)
    cal = load_base_calibration()
    if cal is None:
        print("  SKIP place-error table: no base-frame calibration")
        return None
    pm = gm.load_policy_map()
    keep = set(zip(df.config_id, df.episode_id, df.repeat, df.cube_size))

    recs = []
    for mp in glob.glob(os.path.join(REAL_ROOT, "*", "gap_v1", "*",
                                      "ur3_pick_meta.json")):
        m = json.load(open(mp, encoding="utf-8"))
        if pm.get(m["config_id"]) != m.get("policy_run_id"):
            continue
        if m.get("stop_reason") != "complete" or m.get("landed_box_xy") is None:
            continue
        key = (m["config_id"], m["episode_id"], m["repeat"], m.get("cube_size"))
        if key not in keep:
            continue   # superseded re-run, dropped by the same rule as the CSV
        d = os.path.dirname(mp)
        mi = json.load(open(os.path.join(d, "measured_init.json"), encoding="utf-8"))
        z = float(mi["box_pos"][2])
        pb = mocap_pos_to_base([m["landed_box_xy"][0], m["landed_box_xy"][1], z], cal)
        err, ins = gm.place_error((pb[0], pb[1]))
        recs.append(dict(config=m["config_id"], cube=m.get("cube_size"),
                         episode_id=m["episode_id"], repeat=m["repeat"],
                         place_error_logged_m=m.get("place_error_m"),
                         place_error_m=err, in_square=ins))
    d = pd.DataFrame(recs)
    if d.empty:
        print("  SKIP place-error table: no scored trials")
        return None

    rows, agg = [], []
    for c in LEVELS:
        sub = d[(d.config == c) & (d.cube == PRIMARY_CUBE)]
        if not len(sub):
            continue
        n_in = int(sub.in_square.sum())
        agg.append(dict(config=c, n=len(sub), n_in_square=n_in,
                        median_error_m=sub.place_error_m.median(),
                        best_error_m=sub.place_error_m.min()))
        rows.append([PRETTY[c], str(len(sub)), f"{n_in}",
                     _fmt(100 * sub.place_error_m.median(), 1, False),
                     _fmt(100 * sub.place_error_m.min(), 1, False)])
        _rec(f"place_in_square_{c}", f"{n_in}/{len(sub)}",
             "trials landing inside the 15x15 cm square, 3 cm cube")
        _rec(f"place_best_cm_{c}", round(100 * float(sub.place_error_m.min()), 1),
             "closest landing, 3 cm cube")
    a = pd.DataFrame(agg)
    tex = _latex_table(
        "Place error on the 3\\,cm cube for completed trials: distance from "
        "where the cube landed after the scripted gripper release to the "
        "centre of the $15 \\times 15$\\,cm target square (half-width "
        "7.5\\,cm). Recomputed from the logged landing position with the "
        "base-frame calibration applied; the value written by the campaign "
        "script is in the wrong frame and is not used.",
        "tab:place_error", "lrrrr",
        ["Configuration", "$n$", "In square", "Median err.\\ (cm)",
         "Best (cm)"], rows,
        note="Real-only: MJX has no scripted release event, so there is no "
             "simulated counterpart and this is never folded into the "
             "reward-based metrics.")
    _write("t6_place_error", tex, d)
    a.to_csv(os.path.join(TABLES_DIR, "t6_place_error_summary.csv"), index=False)
    return d


# ===========================================================================
# T8 -- task outcome in simulation (success flag from the env)
#
# Context for the retention numbers: retention is a RATIO of two returns, and
# a ratio says nothing about whether either system solved the task. The env's
# own success flag shows the simulated policies are themselves far from
# reliable, so the sim-to-real comparison is between two imperfect systems,
# not between a solved simulation and a failing robot.
# ===========================================================================


def t_sim_success(df, cube=PRIMARY_CUBE):
    import glob
    pm = gm.load_policy_map()
    keep = set(zip(df.config_id, df.episode_id, df.repeat, df.cube_size))
    recs = []
    for mp in glob.glob(os.path.join(REPO_ROOT, "robots", "UR3e", "sim_results",
                                      "gap_protocol", "*", "gap_v1", "*",
                                      "sim_meta.json")):
        m = json.load(open(mp, encoding="utf-8"))
        if pm.get(m["config_id"]) != m.get("policy_run_id"):
            continue
        key = (m["config_id"], m["episode_id"], m["repeat"], m.get("cube_size"))
        if key not in keep:
            continue
        recs.append(dict(config=m["config_id"], cube=m.get("cube_size"),
                         success=bool(m.get("success")),
                         final_box_target_dist=m.get("final_box_target_dist")))
    d = pd.DataFrame(recs)
    if d.empty:
        print("  SKIP sim-success table: no sim metadata matched")
        return None
    rows = []
    for c in LEVELS:
        sub = d[(d.config == c) & (d.cube == cube)]
        if not len(sub):
            continue
        ns = int(sub.success.sum())
        rows.append([PRETTY[c], str(len(sub)), str(ns),
                     _fmt(100 * ns / len(sub), 0, False) + "\\,\\%",
                     _fmt(100 * sub.final_box_target_dist.median(), 1, False)])
        _rec(f"sim_success_{c}", f"{ns}/{len(sub)}",
             f"simulated episodes meeting the env success criterion, {cube}")
    tex = _latex_table(
        f"Task outcome in simulation on the {cube.replace('cm','~cm')} cube: "
        "the environment's own success flag over the matched-initialisation "
        "rollouts, and the median final cube-to-target distance. Even in "
        "simulation the policies succeed on a minority of episodes, so the "
        "retention ratios in Table~\\ref{tab:retention} compare two imperfect "
        "systems rather than a solved simulation against a failing robot.",
        "tab:sim_success", "lrrrr",
        ["Configuration", "$n$", "Successes", "Rate", "Median final dist.\\ (cm)"],
        rows)
    _write("t8_sim_success", tex, d)
    return d


# ===========================================================================
# T9 / T10 -- evaluation coverage against the TRAINING distribution.
#
# The board deliberately probes robustness, so several of its conditions sit
# at or beyond the edge of what the policies were trained on. That does not
# invalidate the paired sim-vs-real comparison (both domains are given the
# SAME condition), but it does mean the absolute returns are not an
# achievable ceiling, and it confounds L0 specifically: gen_dr_ladder.py
# collapses every pose parameter to a single point for L0, so the board is
# out of distribution for L0 by construction.
#
# Ranges are READ from the env config and the ladder generator, never
# retyped; the realized tilt/height are read from the sim mirror's metadata.
# ===========================================================================

# episode_id -> board condition, from the sim mirror's condition_id field.
EPISODE_CONDITION = {0: "A1", 1: "A2", 2: "A3", 3: "B1", 4: "B2", 5: "C1",
                     6: "D1", 7: "E1", 8: "E2"}
BASELINE_EPISODES = [0, 1, 2]      # A1-A3: field position only, flat, 3 cm
PERTURBED_EPISODES = [3, 4, 5, 6]  # B yaw, C tilt, D raised support


def _training_ranges():
    """Trained pose ranges, read from the env config and the ladder generator."""
    from mujoco_playground._src.manipulation.my_ur3 import ur3_pick
    c = ur3_pick.default_config()
    out = {
        "box_xy_jitter": tuple(float(v) for v in c.box_xy_jitter),
        "box_z_rot_range": float(c.box_z_rot_range),
        "lifter_tilt_max": float(c.lifter_tilt_max),
        "lifter_height_abs": (float(c.lifter_height_abs_min),
                               float(c.lifter_height_abs_max)),
        "cube_size_xy": (float(c.domain_rand.cube_size_xy.min),
                          float(c.domain_rand.cube_size_xy.max)),
    }
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "batch_runs", "sweeps"))
        import gen_dr_ladder as gl
        out["L0_deterministic"] = dict(gl._DETERMINISTIC_POSITION)
    except Exception as e:  # noqa: BLE001
        out["L0_deterministic"] = f"unavailable ({e})"
    return out


def _realized_condition_values():
    """Tilt/height the sim mirror actually reproduced for C1 / D1."""
    import glob
    tilts, heights = [], []
    for p in glob.glob(os.path.join(REPO_ROOT, "robots", "UR3e", "sim_results",
                                     "gap_protocol", "*", "gap_v1", "C1_*",
                                     "sim_meta.json")):
        v = json.load(open(p, encoding="utf-8")).get("lifter_tilt_rp_rad")
        if v:
            tilts.append(max(abs(float(x)) for x in v))
    for p in glob.glob(os.path.join(REPO_ROOT, "robots", "UR3e", "sim_results",
                                     "gap_protocol", "*", "gap_v1", "D1_*",
                                     "sim_meta.json")):
        v = json.load(open(p, encoding="utf-8")).get("lifter_top_height_m")
        if v is not None:
            heights.append(float(v))
    return tilts, heights


def t_distribution_coverage():
    rng = _training_ranges()
    tilts, heights = _realized_condition_values()
    tilt_max = max(tilts) if tilts else float("nan")
    h_lo = min(heights) if heights else float("nan")
    h_hi = max(heights) if heights else float("nan")
    _rec("trained_tilt_max_rad", rng["lifter_tilt_max"], "lifter_tilt_max")
    _rec("trained_tilt_max_deg", round(math.degrees(rng["lifter_tilt_max"]), 2), "")
    _rec("realized_tilt_max_deg", round(math.degrees(tilt_max), 2) if tilts else None,
         "largest tilt the C1 wedge actually produced")
    _rec("realized_height_D1_m", round(h_hi, 3) if heights else None, "")

    tilt_ood = tilts and tilt_max > rng["lifter_tilt_max"]
    rows = [
        ["A1--A3", "0--2", "Field position (P1/P2/P3)",
         f"\\texttt{{box\\_xy\\_jitter}} {rng['box_xy_jitter']}", "in", "\\textbf{out}"],
        ["B1--B2", "3--4", "Cube yaw $45^\\circ$",
         f"\\texttt{{box\\_z\\_rot\\_range}} ${rng['box_z_rot_range']:.2f}$ rad (full)",
         "in", "\\textbf{out}"],
        ["C1", "5", f"Table tilt {math.degrees(tilt_max):.1f}$^\\circ$" if tilts
         else "Table tilt",
         f"\\texttt{{lifter\\_tilt\\_max}} {math.degrees(rng['lifter_tilt_max']):.1f}$^\\circ$",
         "\\textbf{out}" if tilt_ood else "in", "\\textbf{out}"],
        ["D1", "6", f"Support height {h_hi*1000:.0f}~mm" if heights
         else "Raised support",
         f"\\texttt{{lifter\\_height\\_abs}} {rng['lifter_height_abs'][0]:.2f}--{rng['lifter_height_abs'][1]:.2f}~m",
         "in", "\\textbf{out}"],
        ["E1--E2", "7--8", "4~cm cube",
         f"\\texttt{{cube\\_size\\_xy}} $\\leq{rng['cube_size_xy'][1]:.3f}$~m half-extent",
         "\\textbf{out}", "\\textbf{out}"],
    ]
    tex = _latex_table(
        "Board conditions against the pose ranges seen during training. "
        "\\enquote{in} means the condition falls inside the randomization "
        "range that configuration was trained with. $L0$ is out of "
        "distribution everywhere by construction: the ladder generator "
        "collapses every pose parameter to a single point for it, so it saw "
        "one cube position, one yaw, a flat table at nominal height and one "
        "arm pose.",
        "tab:coverage", "llllcc",
        ["Cond.", "Ep.", "What varies", "Trained range", "$L1$--$L4$", "$L0$"],
        rows,
        note="Ranges are read from \\texttt{ur3\\_pick.default\\_config()} and "
             "\\texttt{batch\\_runs/sweeps/gen\\_dr\\_ladder.py}; the realized "
             "tilt and height are read from the sim mirror's metadata, which "
             "reproduces the physically measured condition. The paired "
             "comparison remains valid for out-of-distribution conditions, "
             "since simulation is given the same condition as the robot; what "
             "they do not support is reading the absolute returns as an "
             "achievable ceiling.")
    d = pd.DataFrame(dict(condition=["A1-A3", "B1-B2", "C1", "D1", "E1-E2"],
                          episodes=["0-2", "3-4", "5", "6", "7-8"],
                          in_dist_L1_L4=["in", "in",
                                         "out" if tilt_ood else "in", "in", "out"],
                          in_dist_L0=["out"] * 5))
    _write("t9_coverage", tex, d)
    with open(os.path.join(TABLES_DIR, "t9_training_ranges.json"), "w") as f:
        json.dump({"trained": rng, "realized_tilt_rad": tilts,
                   "realized_height_m": heights}, f, indent=2, default=str)
    return rng


def t_condition_split(df, cube=PRIMARY_CUBE):
    """Retention on baseline conditions vs perturbed ones, to show whether the
    cross-configuration comparison is driven by the out-of-distribution mix."""
    rows, recs = [], []
    for c in LEVELS:
        r_b = gm.retention(df, c, cube_size=cube, episodes=BASELINE_EPISODES,
                            n_boot=N_BOOT, seed=BOOT_SEED,
                            terms=gm.EXACT_PARITY_TERMS)
        r_p = gm.retention(df, c, cube_size=cube, episodes=PERTURBED_EPISODES,
                            n_boot=N_BOOT, seed=BOOT_SEED,
                            terms=gm.EXACT_PARITY_TERMS)
        recs.append(dict(config=c, n_baseline=r_b["n_episodes"],
                         retention_baseline=r_b["retention"],
                         R_real_baseline=r_b["R_real"],
                         n_perturbed=r_p["n_episodes"],
                         retention_perturbed=r_p["retention"],
                         R_real_perturbed=r_p["R_real"]))
        rows.append([PRETTY[c], str(r_b["n_episodes"]),
                     _fmt(r_b["retention"], 2, False), str(r_p["n_episodes"]),
                     _fmt(r_p["retention"], 2, False)])
        _rec(f"retention_baseline_{c}",
             None if not np.isfinite(r_b["retention"]) else round(float(r_b["retention"]), 3),
             "retention on A1-A3 only")
        _rec(f"retention_perturbed_{c}",
             None if not np.isfinite(r_p["retention"]) else round(float(r_p["retention"]), 3),
             "retention on B1,B2,C1,D1")
    d = pd.DataFrame(recs)
    tex = _latex_table(
        "Retention split by condition group on the 3~cm cube: the baseline "
        "conditions A1--A3, which vary only the field position, against the "
        "perturbed conditions B1, B2, C1 and D1. For the four configurations "
        "that learn the task the two agree within the sampling error, so the "
        "comparison between configurations is not an artifact of the "
        "perturbed conditions. $L0$ has a single paired baseline episode and "
        "its split is not interpretable.",
        "tab:condition_split", "lrrrr",
        ["Configuration", "$n$ base", "Retention base", "$n$ pert.",
         "Retention pert."], rows,
        note="Most configurations contribute only one to three baseline "
             "episodes, so the retention headline in "
             "Table~\\ref{tab:retention} is dominated by the perturbed "
             "conditions.")
    _write("t10_condition_split", tex, d)
    return d


# ===========================================================================
# T7 -- sim-selection regret
# ===========================================================================


def t_regret(df, cube=PRIMARY_CUBE):
    res = gm.sim_selection_regret(df, configs=LEVELS, cube_size=cube,
                                   n_boot=N_BOOT, seed=BOOT_SEED,
                                   terms=gm.EXACT_PARITY_TERMS)
    t = res["table"]
    rows = [[PRETTY[r.config_id], _fmt(r.R_sim, 0), _fmt(r.R_real, 0),
             _fmt(100 * res["selection_freq"][r.config_id], 0, False) + "\\,\\%"]
            for r in t.itertuples()]
    _rec("regret", round(float(res["regret"]), 1), "sim-selection regret")
    _rec("regret_ci_hi", round(float(res["ci95_resampled_argmax"][1]), 1), "")
    _rec("regret_n_common_episodes", int(res["n_episodes"]),
         "episodes shared by all five configurations, both domains")
    _rec("regret_pick_sim", res["chat_sim"], "")
    _rec("regret_pick_real", res["c_star"], "")
    tex = _latex_table(
        "Sim-selection regret on the "
        f"{cube.replace('cm','~cm')} cube: the real return given up by "
        "choosing a configuration on simulated return alone. "
        "\\enquote{Picked} is how often the bootstrap selects that "
        "configuration as the simulated best.",
        "tab:regret", "lrrr",
        ["Configuration", "$R_{\\mathrm{sim}}$", "$R_{\\mathrm{real}}$",
         "Picked"], rows,
        note=(f"Regret {res['regret']:.0f} "
              f"(95\\,\\% CI [{res['ci95_resampled_argmax'][0]:.0f}, "
              f"{res['ci95_resampled_argmax'][1]:.0f}]) over the "
              f"{res['n_episodes']} episodes common to all five "
              f"configurations in both domains. The point estimate is zero "
              f"because simulation and reality agree on the best "
              f"configuration here, but with only {res['n_episodes']} shared "
              f"episodes the interval still admits a substantial loss."))
    d = t.copy()
    d["selection_freq"] = d.config_id.map(res["selection_freq"])
    d["regret"] = res["regret"]
    d["regret_ci_lo"] = res["ci95_resampled_argmax"][0]
    d["regret_ci_hi"] = res["ci95_resampled_argmax"][1]
    d["n_common_episodes"] = res["n_episodes"]
    _write("t7_regret", tex, d)
    return res


# ===========================================================================
# Figures
# ===========================================================================

FIGURE_RENAMES = {
    "f1_dose_response": "f1-dose-response",
    "f2_term_retention": "f2-term-retention",
    "f5_selection_regret": "f5-selection-regret",
    "f9_abort_rate_bar": "f9-abort-rate",
    "f6_replay_error": "f6-replay-error",
    "f6b_replay_aggregate": "f6b-replay-aggregate",
    "f4_paired_scatter_L2_pos_cube": "f4-paired-scatter-l2",
}


def build_figures():
    """Regenerate every report figure and copy it under its report name."""
    import shutil
    from evaluation import run_gap_plots
    os.makedirs(FIGURES_DIR, exist_ok=True)
    src = os.path.join(EVAL_DIR, "gap_plots")
    print("\n[figures] regenerating from gap_metrics.csv ...")
    run_gap_plots.run(CSV_PATH, src)
    n = 0
    for stem, newstem in FIGURE_RENAMES.items():
        for ext in ("pdf", "png"):
            s = os.path.join(src, f"{stem}.{ext}")
            if os.path.exists(s):
                shutil.copyfile(s, os.path.join(FIGURES_DIR, f"{newstem}.{ext}"))
                n += 1
    print(f"[figures] copied {n} files into {os.path.relpath(FIGURES_DIR, REPO_ROOT)}")
    missing = [k for k in FIGURE_RENAMES
               if not os.path.exists(os.path.join(src, k + ".pdf"))]
    if missing:
        print(f"[figures] NOT regenerated here (run evaluation/run_f6_replay.py "
              f"for the replay figures): {missing}")


# ===========================================================================


def main(tables_only=False):
    os.makedirs(TABLES_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"{CSV_PATH} missing -- run: "
                          f"python evaluation/gap_metrics.py --build")
    print(f"[report] input {os.path.relpath(CSV_PATH, REPO_ROOT)}")
    df = gm.load_long(CSV_PATH)
    _assert_d23_only(df)

    print("\n[tables]")
    t_dataset(df)
    t_retention(df)
    t_stage_retention(df)
    t_abort(df)
    t_replay()
    t_place_error(df)
    t_sim_success(df)
    t_regret(df)
    t_distribution_coverage()
    t_condition_split(df)
    _write_bundle()

    with open(os.path.join(_THIS_DIR, "report_numbers.json"), "w",
              encoding="utf-8") as f:
        json.dump(NUMBERS, f, indent=2, sort_keys=True, default=str)
    print(f"  wrote report_numbers.json ({len(NUMBERS)} values)")

    if not tables_only:
        build_figures()

    prov = [
        "VT2 Results -- provenance",
        "=" * 60,
        f"git commit      : {_git('rev-parse', 'HEAD')}",
        f"git branch      : {_git('rev-parse', '--abbrev-ref', 'HEAD')}",
        f"git status      : {'dirty' if _git('status','--porcelain') else 'clean'}",
        f"gap_metrics.csv : sha256 {_sha256(CSV_PATH)}",
        f"                  {len(df)} rows",
    ]
    if os.path.exists(F6_CSV_PATH):
        prov.append(f"f6_replay_summary.csv : sha256 {_sha256(F6_CSV_PATH)}")
    prov += [
        "",
        "Regenerate everything from the raw run folders:",
        "  python evaluation/gap_metrics.py --build      # -> gap_metrics.csv",
        "  python evaluation/run_f6_replay.py            # -> f6_replay_summary.csv",
        "  python evaluation/report/build_report.py      # -> tables/ figures/",
    ]
    with open(os.path.join(_THIS_DIR, "PROVENANCE.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(prov) + "\n")
    print("\n" + "\n".join(prov))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", action="store_true", help="tables only, skip figures")
    a = ap.parse_args()
    main(tables_only=a.tables)
