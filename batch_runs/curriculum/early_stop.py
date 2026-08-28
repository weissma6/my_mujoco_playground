"""Per-rung early stopping for the curriculum ladder.

Pure Python on purpose: no jax, no brax, no wandb, no mujoco. This module is the
one piece of the curriculum that must be fully testable on the MacBook, where
MJX may never run.

WHY AN EXCEPTION IS NEEDED TO ACT ON THIS
-----------------------------------------
brax 0.13.0's PPO trainer runs a fixed budget and asserts it was consumed --
`brax/training/agents/ppo/train.py:750-755` raises AssertionError when
`total_steps < num_timesteps` -- and it reads the return value of neither
`progress_fn` nor `policy_params_fn`. There is no "stop here" hook. Raising
`ConvergedSignal` from inside `progress_fn` is the only way out of that loop.

That is safe because of the callback ordering inside one eval iteration:

    train.py:727   policy_params_fn(current_step, make_policy, params)
    train.py:748   progress_fn(current_step, metrics)

Both fire at the same `current_step`, and `metrics` are the result of evaluating
exactly those `params` (train.py:742-745). So when the tracker decides to stop at
eval k, the params already stashed by `policy_params_fn` are the ones that earned
the score being judged.

DIVISION OF LABOUR
------------------
`PatienceTracker.update()` returns a bool and never raises. The caller decides
whether to turn that into control flow. That keeps the stop *policy* testable
without exception plumbing, and keeps the *mechanism* (ConvergedSignal) in the
one place that talks to brax.
"""

from typing import Any, Dict, List, Optional, Tuple

__all__ = ["ConvergedSignal", "PatienceTracker"]


class ConvergedSignal(Exception):
    """Raised from progress_fn to escape brax's fixed-budget eval loop.

    Carries the BEST observed eval, not the last one -- the last eval is by
    definition a non-improvement (that is why we are stopping), so reporting it
    would understate what the rung achieved.
    """

    def __init__(
        self,
        step: int,
        best_reward: float,
        best_step: int,
        reason: str,
    ) -> None:
        self.step = int(step)
        self.best_reward = float(best_reward)
        self.best_step = int(best_step)
        self.reason = str(reason)
        super().__init__(
            f"converged at step {self.step}: {self.reason} "
            f"(best {self.best_reward:.4g} at step {self.best_step})"
        )


class PatienceTracker:
    """Relative-improvement patience on a scalar eval metric.

    An eval counts as an improvement when

        reward > best + abs(best) * min_delta

    One formula covers both signs. For ``best > 0`` it is exactly
    ``best * (1 + min_delta)``; for ``best < 0`` it correctly asks the reward to
    move *up* by ``min_delta`` of the magnitude (best=-100, min_delta=0.02 ->
    threshold -98). Writing it as ``best * (1 + min_delta)`` would invert in the
    negative regime and demand the reward get *worse*, which is why the
    magnitude form is used instead. UR3Pick reward is positive in practice, but
    it starts near zero and a sign flip early in training must not silently
    disable the stop rule.

    `patience` consecutive non-improvements past the `min_steps` floor -> stop.
    The floor exists because UR3Pick reliably only learns reach/grasp by ~5M
    steps; without it a flat early curve would stop every rung before it starts.

    `count_stale_before_floor=False` (the default) additionally refuses to even
    COUNT non-improvements below the floor. Without that the floor only delays
    the verdict rather than deferring the evidence: a rung whose first seven
    evals are flat -- the normal shape -- would cross the floor with patience
    already spent and stop at exactly `min_steps`. See the test of the same
    name.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.02,
        min_steps: int = 6_000_000,
        metric: str = "eval/episode_reward",
        count_stale_before_floor: bool = False,
    ) -> None:
        if int(patience) < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if float(min_delta) < 0.0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        if int(min_steps) < 0:
            raise ValueError(f"min_steps must be >= 0, got {min_steps}")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_steps = int(min_steps)
        self.metric = str(metric)
        self.count_stale_before_floor = bool(count_stale_before_floor)

        # The bar: updated only on a *significant* improvement (Keras
        # EarlyStopping semantics). A creeping sub-min_delta rise must not keep
        # resetting patience, or a flat-but-noisy tail never stops.
        self.best_reward: Optional[float] = None
        self.best_step: Optional[int] = None

        # The true running max, for reporting. Diverges from the bar whenever a
        # rise was real but smaller than min_delta.
        self.max_reward: Optional[float] = None
        self.max_step: Optional[int] = None

        self.strikes = 0
        self.stopped = False
        self.reason: Optional[str] = None
        self.history: List[Tuple[int, float]] = []

    def improvement_threshold(self) -> float:
        """The value an eval must exceed to reset patience."""
        if self.best_reward is None:
            return float("-inf")
        return self.best_reward + abs(self.best_reward) * self.min_delta

    def update(self, step: int, metrics: Dict[str, Any]) -> bool:
        """Feed one eval. Return True when the rung should stop.

        Never returns True before `min_steps`. Never raises -- the caller turns
        a True into a ConvergedSignal.
        """
        if self.metric not in metrics:
            return False
        try:
            reward = float(metrics[self.metric])
        except (TypeError, ValueError):
            return False

        step = int(step)
        self.history.append((step, reward))

        if self.max_reward is None or reward > self.max_reward:
            self.max_reward = reward
            self.max_step = step

        if reward > self.improvement_threshold():
            self.best_reward = reward
            self.best_step = step
            self.strikes = 0
        else:
            # Only accrue staleness at or past the floor (default). Otherwise a
            # flat early curve -- which is the NORMAL shape here, UR3Pick only
            # learns reach/grasp by ~5M steps -- arrives at the floor with
            # patience already spent and the rung dies at exactly min_steps.
            # At num_evals=30 over 24M (~828k/eval) the 6M floor is eval ~7, so
            # a patience of 5 would be exhausted before training had begun.
            # Gating guarantees min_steps + patience*eval_interval instead.
            if self.count_stale_before_floor or step >= self.min_steps:
                self.strikes += 1

        if self.strikes >= self.patience and step >= self.min_steps:
            self.stopped = True
            self.reason = (
                f"{self.strikes} consecutive evals without a "
                f">{self.min_delta:.1%} improvement"
            )
            return True
        return False

    def signal(self, step: int) -> ConvergedSignal:
        """Build the ConvergedSignal for a stop decision at `step`."""
        return ConvergedSignal(
            step=step,
            best_reward=self.max_reward if self.max_reward is not None else float("nan"),
            best_step=self.max_step if self.max_step is not None else -1,
            reason=self.reason or "converged",
        )

    def summary(self) -> Dict[str, Any]:
        """JSON-safe record of what the tracker saw. Logged with the rung."""
        return {
            "metric": self.metric,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_steps": self.min_steps,
            "count_stale_before_floor": self.count_stale_before_floor,
            "stopped": self.stopped,
            "reason": self.reason,
            "strikes": self.strikes,
            "best_reward": self.max_reward,
            "best_step": self.max_step,
            "n_evals": len(self.history),
            "last_step": self.history[-1][0] if self.history else None,
        }
