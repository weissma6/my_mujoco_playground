"""Pair each eval's params with the reward measured on THOSE params.

THE BUG THIS EXISTS TO FIX
--------------------------
`run_experiment.py` used to choose the best checkpoint inside
`policy_params_fn`:

    _r = prog_state.get("last_eval_reward")        # written by progress_fn
    if _r is not None and _r > best_ckpt["reward"]:
        best_ckpt.update(reward=float(_r), step=int(num_steps), params=params)

but brax fires the two callbacks in this order (ppo/train.py):

    :689   progress_fn(0, metrics_0)          # initial eval
    :697   policy_params_fn(0, ...)
    loop -> :727 policy_params_fn(s_k, ...)
            :748 progress_fn(s_k, metrics_k)

so from the first loop iteration onward `policy_params_fn(s_k)` reads the
reward from eval *k-1*. Every params tree was scored with its predecessor's
number, and `trained_policy/best_params.msgpack` therefore held the params from
**one eval past the peak** -- on a curve that peaks and then collapses (exactly
what `run_experiment.py`'s own comment says UR3Pick does) that is the worst
possible thing to publish as "best".

THE FIX
-------
Match on `step`, which both callbacks receive, rather than on arrival order.
The recorder is then correct under either ordering -- which matters because the
order is brax's to change, not ours, and because a step-0 special case
("progress happens to come first there") is exactly the kind of coincidence
that quietly stops holding.
"""

import os
from typing import Any, Callable, Dict, Optional, Tuple

__all__ = ["BestParamsRecorder"]


class BestParamsRecorder:
    """Tracks the best (params, reward) pair and snapshots every eval to disk.

    `on_policy_params` and `on_progress` may arrive in either order for a given
    step; the pair is resolved as soon as both halves are present.
    """

    def __init__(
        self,
        ckpt_dir: Optional[str] = None,
        serialize: Optional[Callable[[Any], bytes]] = None,
        on_error: Optional[Callable[[int, Exception], None]] = None,
    ) -> None:
        self.ckpt_dir = ckpt_dir
        self._serialize = serialize
        self._on_error = on_error

        self.best: Dict[str, Any] = {"reward": float("-inf"), "step": -1, "params": None}
        self.latest: Dict[str, Any] = {"step": -1, "params": None}

        self._pending_params: Dict[int, Any] = {}
        self._pending_rewards: Dict[int, float] = {}
        self.n_snapshots = 0

    # -- callbacks ---------------------------------------------------------

    def on_policy_params(self, step: int, params: Any) -> None:
        step = int(step)
        self.latest = {"step": step, "params": params}
        self._pending_params[step] = params
        self._snapshot(step, params)
        self._resolve(step)

    def on_progress(self, step: int, reward: Optional[float]) -> None:
        if reward is None:
            return
        try:
            reward = float(reward)
        except (TypeError, ValueError):
            return
        step = int(step)
        self._pending_rewards[step] = reward
        self._resolve(step)

    # -- internals ---------------------------------------------------------

    def _resolve(self, step: int) -> None:
        if step not in self._pending_params or step not in self._pending_rewards:
            return
        reward = self._pending_rewards.pop(step)
        params = self._pending_params.pop(step)
        if reward > self.best["reward"]:
            self.best = {"reward": reward, "step": step, "params": params}

    def _snapshot(self, step: int, params: Any) -> None:
        if self.ckpt_dir is None or self._serialize is None:
            return
        try:
            path = os.path.join(self.ckpt_dir, f"params_step{step:09d}.msgpack")
            with open(path, "wb") as f:
                f.write(self._serialize(params))
            self.n_snapshots += 1
        except Exception as e:  # noqa: BLE001 - snapshotting must never kill a run
            if self._on_error is not None:
                self._on_error(step, e)
            else:
                print(f"[ckpt] save failed at step {step}: {e}", flush=True)

    # -- reporting ---------------------------------------------------------

    def best_info(self) -> Dict[str, Any]:
        return {"best_eval_reward": self.best["reward"], "best_step": self.best["step"]}

    def unpaired(self) -> Tuple[int, int]:
        """(params without a reward, rewards without params) -- diagnostics."""
        return len(self._pending_params), len(self._pending_rewards)
