"""Back-compat shim.

The PPO training entry point is now the robot-independent
`learning/notebooks/run_experiment.py`. This module re-exports it (and its
public helpers) so existing imports keep working with no duplicated logic.
"""

from learning.notebooks.run_experiment import *  # noqa: F401,F403
from learning.notebooks.run_experiment import run_experiment  # noqa: F401
