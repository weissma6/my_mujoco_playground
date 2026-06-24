"""Histogram view of a UR3e calibration repeatability report.

Reads an `accuracy_report_<ts>.csv` (or the accumulating `calibration_history.csv`
-- both share the 22-column schema written by base_calibration.append_run_record)
and draws the run-to-run distributions:

  * Row 1 -- per-axis TCP position (X/Y/Z, mm): the RTDE TCP `tcp_world`
    ("from receive", 1st-sweep calibration) overlaid with the geometric forward
    TCP `tcp_forward` (from the wrist / 2nd sweep). Shared bins, so the per-axis
    spread AND the world<->forward offset (e.g. ~1.6 mm on Z) are both visible.
  * Row 2 -- |tcp_world - tcp_forward| (the sanity residual) and
    |tcp_world - mocap| (the fixed calibrig4->TCP mount offset, ~47 mm).
  * A text panel with the numbers.

Hardware-free (matplotlib + numpy + stdlib csv): replots any old report and never
calls plt.show() (so it is safe headless; savefig works on any backend). Pass
`--show` to also open a window.

    cd robots/UR3e/calibration/calibration_accuracy
    python plot_accuracy.py                                   # latest report
    python plot_accuracy.py --csv calibration_history.csv     # all runs (smoother)
    python plot_accuracy.py --csv accuracy_report_<ts>.csv --out fig.png

Note: a single batch is N~10 runs, so the histograms are coarse (~5 bins). Point
it at calibration_history.csv to pool every run for a smoother picture.
"""

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# RTDE TCP (mapped to world via p0/R0) vs the geometric forward TCP (wrist axes).
_C_WORLD, _C_FWD = "C0", "C3"


def load_report(csv_path):
    """Read a report CSV -> (cols, n).

    cols maps each numeric column name to a float np.ndarray; blank cells (e.g. a
    run whose forward check was skipped) become np.nan. The `timestamp` column is
    dropped. n is the number of data rows.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no data rows in {csv_path}")
    cols = {}
    for key in rows[0]:
        if key == "timestamp":
            continue
        cols[key] = np.array(
            [float(r[key]) if r[key].strip() != "" else np.nan for r in rows]
        )
    return cols, len(rows)


def _rug(ax, vals, color):
    """Thin tick per sample just below the x-axis (legible even with ~10 runs)."""
    ax.plot(vals, np.full_like(vals, -0.06), "|", color=color, ms=8,
            transform=ax.get_xaxis_transform(), clip_on=False)


def _hist_overlay(ax, a, b, labels, unit):
    """Two overlaid histograms on shared bins + per-series mean line + rug."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    bins = np.histogram_bin_edges(np.concatenate([a, b]), bins="auto")
    for arr, col, lab in ((a, _C_WORLD, labels[0]), (b, _C_FWD, labels[1])):
        ax.hist(arr, bins=bins, alpha=0.55, color=col, label=lab)
        ax.axvline(arr.mean(), color=col, ls="--", lw=1.2)
        _rug(ax, arr, col)
    ax.set_xlabel(unit)
    ax.legend(fontsize=7)


def _hist_single(ax, a, unit, color):
    """One histogram + mean/std line + rug, NaNs dropped."""
    a = a[~np.isnan(a)]
    ax.hist(a, bins="auto", color=color, alpha=0.8)
    ax.axvline(a.mean(), color="k", ls="--", lw=1.2,
               label=f"mean {a.mean():.2f}\nstd  {a.std():.2f}")
    _rug(ax, a, "k")
    ax.set_xlabel(unit)
    ax.legend(fontsize=7)


def _summary_text(cols, n, source):
    """Monospace stats block for the 6th panel."""

    def ms(arr):
        arr = arr[~np.isnan(arr)]
        return arr.mean(), arr.std()

    lines = [f"N = {n} runs", f"source: {source}", "",
             "TCP position (mm)   tcp_world      tcp_forward"]
    for ax_name in "xyz":
        wm, ws = ms(cols[f"tcp_world_{ax_name}"] * 1e3)
        fm, fs = ms(cols[f"tcp_forward_{ax_name}"] * 1e3)
        lines.append(f"  {ax_name.upper()}   {wm:9.2f}+-{ws:4.2f}  {fm:9.2f}+-{fs:4.2f}")
    d = cols["dist_world_to_forward_mm"]
    d = d[~np.isnan(d)]
    dw = cols["d_world_mm"]
    dw = dw[~np.isnan(dw)]
    lines += [
        "",
        f"|world-forward| = {d.mean():5.2f} +- {d.std():4.2f} mm",
        f"                  ({d.min():.2f} .. {d.max():.2f})",
        f"|world-mocap|   = {dw.mean():5.2f} +- {dw.std():4.2f} mm",
        "                  (calibrig4->TCP mount offset)",
    ]
    n_fwd = int(np.sum(~np.isnan(cols["tcp_forward_x"])))
    if n_fwd < n:
        lines += ["", f"note: {n - n_fwd}/{n} runs had no forward",
                  "      TCP (dropped from those panels)"]
    return "\n".join(lines)


def plot_histograms(cols, n, out_png, source=""):
    """Build the 2x3 repeatability figure and save it to out_png."""
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3)

    labels = ("RTDE tcp_world (1st sweep)", "forward tcp (2nd sweep)")
    for j, ax_name in enumerate("xyz"):
        ax = fig.add_subplot(gs[0, j])
        _hist_overlay(ax, cols[f"tcp_world_{ax_name}"] * 1e3,
                      cols[f"tcp_forward_{ax_name}"] * 1e3,
                      labels, f"{ax_name.upper()} (mm)")
        ax.set_title(f"TCP {ax_name.upper()}  (world vs forward)")

    ax = fig.add_subplot(gs[1, 0])
    _hist_single(ax, cols["dist_world_to_forward_mm"], "mm", "C2")
    ax.set_title("|tcp_world - tcp_forward|  (sanity residual)")

    ax = fig.add_subplot(gs[1, 1])
    _hist_single(ax, cols["d_world_mm"], "mm", "C1")
    ax.set_title("|tcp_world - mocap|  (mount offset)")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(0.0, 1.0, _summary_text(cols, n, source), va="top",
            family="monospace", fontsize=8.5)

    fig.suptitle(f"UR3e calibration repeatability  (N={n})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("wrote", out_png)
    return fig


def _default_csv():
    reports = sorted(glob.glob(os.path.join(_HERE, "accuracy_report_*.csv")))
    if not reports:
        raise FileNotFoundError(
            f"no accuracy_report_*.csv in {_HERE}; pass --csv explicitly")
    return reports[-1]  # timestamped names sort chronologically


def _default_out(csv_path):
    base = os.path.basename(csv_path)
    if base.startswith("accuracy_report_"):
        base = base.replace("accuracy_report_", "accuracy_histograms_", 1)
        stem = os.path.splitext(base)[0]
    else:
        stem = os.path.splitext(base)[0] + "_histograms"
    return os.path.join(os.path.dirname(os.path.abspath(csv_path)), stem + ".png")


def main():
    ap = argparse.ArgumentParser(description="Histogram a calibration report CSV")
    ap.add_argument("--csv", default=None,
                    help="report CSV (default: latest accuracy_report_*.csv here)")
    ap.add_argument("--out", default=None, help="output PNG (default: derived)")
    ap.add_argument("--show", action="store_true", help="also open a window")
    args = ap.parse_args()

    csv_path = args.csv or _default_csv()
    out_png = args.out or _default_out(csv_path)
    cols, n = load_report(csv_path)
    plot_histograms(cols, n, out_png, source=os.path.basename(csv_path))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
