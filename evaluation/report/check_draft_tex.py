"""Lint the LaTeX inside the thesis draft Markdown files before pasting into
Overleaf.

A missing \\input target, a stray `_` in text mode, or an unbalanced
environment each produce NO PDF AT ALL on Overleaf, with a log that does not
obviously point at the cause. This catches those locally.

    python evaluation/report/check_draft_tex.py [FILE.md ...]

With no arguments it checks the VT2 Results and Conclusions drafts.
"""

import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.report.build_report import lint_tex  # noqa: E402

PROJECT_DIR = os.path.expanduser(
    "~/dev/second_brain_vault/02 Projects/VT2-SimToReal-Robotics")
VAULT = os.path.join(PROJECT_DIR, "Drafts")
DEFAULT_FILES = [os.path.join(VAULT, "4 Results.md"),
                 os.path.join(VAULT, "5 Conclusions.md")]

# Every folder an \includegraphics target might actually live in. The report
# figures (F1, F2, ...) land in evaluation/report/figures/; hand-taken photos
# and other one-off figures land directly in the vault's Figures_final/ (see
# Method.md's table-setup photos) -- checking only the former produced false
# "missing figure" warnings for the latter.
FIGURE_SEARCH_DIRS = [
    os.path.join(_THIS_DIR, "figures"),
    os.path.join(PROJECT_DIR, "Figures", "Figures_final"),
]

ENVIRONMENTS = ["table", "tabular", "figure", "minipage", "itemize",
                "enumerate", "equation", "align"]


def check(path):
    name = os.path.basename(path)
    if not os.path.exists(path):
        return [f"{name}: file not found"]
    src = open(path, encoding="utf-8").read()
    blocks = re.findall(r"```latex\n(.*?)\n```", src, re.S)
    if not blocks:
        return [f"{name}: no ```latex fence found"]
    problems = []
    for bi, tex in enumerate(blocks, 1):
        tag = name if len(blocks) == 1 else f"{name}[block {bi}]"
        problems += lint_tex(tag, tex)

        for env in ENVIRONMENTS:
            nb = len(re.findall(r"\\begin\{" + env + r"\}", tex))
            ne = len(re.findall(r"\\end\{" + env + r"\}", tex))
            if nb != ne:
                problems.append(f"{tag}: {env} unbalanced "
                                f"({nb} begin, {ne} end)")

        # \input targets must exist relative to an Overleaf project; we cannot
        # verify that here, so flag them -- a missing target is fatal.
        for m in re.findall(r"\\input\{([^}]*)\}", tex):
            problems.append(
                f"{tag}: \\input{{{m}}} -- the target must exist in the "
                f"Overleaf project or the compile produces no PDF. Inline the "
                f"table instead, or upload evaluation/report/tables/.")

        # figures referenced must exist SOMEWHERE they'd plausibly be copied
        # from (report figures dir, or the vault's Figures_final/)
        for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
            base = os.path.basename(m)
            if not any(os.path.exists(os.path.join(d, base))
                       for d in FIGURE_SEARCH_DIRS):
                searched = ", ".join(os.path.relpath(d, PROJECT_DIR)
                                     for d in FIGURE_SEARCH_DIRS)
                problems.append(f"{tag}: figure {base} not found in any of: "
                                f"{searched}")

        # undefined-reference hygiene: labels defined here vs referenced here
        labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
        refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
        external = {r for r in refs - labels}
        if external:
            problems.append(f"{tag}: NOTE refs resolved elsewhere (warning, "
                            f"not fatal): {sorted(external)}")
    return problems


def main(argv):
    files = argv[1:] or DEFAULT_FILES
    total_fatal = 0
    for p in files:
        probs = check(p)
        fatal = [x for x in probs if "NOTE" not in x]
        total_fatal += len(fatal)
        print(f"\n{os.path.basename(p)}: {len(fatal)} problem(s)")
        for x in probs:
            print(f"  {'!!' if 'NOTE' not in x else '--'} {x}")
    print()
    if total_fatal:
        print(f"{total_fatal} problem(s) would likely break the Overleaf compile.")
        return 1
    print("Draft LaTeX looks compile-safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
