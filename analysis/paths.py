"""Repository paths shared by the figure and baseline scripts.

`FIGURES_DIR` and `SUPPLEMENTARY_DIR` are resolved rather than fixed: both sit
under `manuscript/` in this tree and at the top level in the published artifact,
which carries no `manuscript/`. Writing to a fixed `manuscript/` path would
create that directory in the artifact instead of failing, so the fallback is
what keeps the published tree free of one. `RESULTS_DIR` honours `RESULTS_ROOT`
so a verification run can regenerate into a scratch tree and diff it against the
committed CSVs.
"""
import os

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

RESULTS_DIR = os.environ.get("RESULTS_ROOT") or os.path.join(REPO_ROOT, "results")

_MANUSCRIPT = os.path.join(REPO_ROOT, "manuscript")

_MANUSCRIPT_FIGURES = os.path.join(_MANUSCRIPT, "figures")
FIGURES_DIR = (
    _MANUSCRIPT_FIGURES
    if os.path.isdir(_MANUSCRIPT_FIGURES)
    else os.path.join(REPO_ROOT, "figures")
)

# Keyed on manuscript/ rather than on the subdirectory itself: unlike
# manuscript/figures/, this one is created on demand and need not exist yet.
SUPPLEMENTARY_DIR = (
    os.path.join(_MANUSCRIPT, "supplementary")
    if os.path.isdir(_MANUSCRIPT)
    else os.path.join(REPO_ROOT, "supplementary")
)
