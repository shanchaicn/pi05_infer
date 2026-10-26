import copy
import threading
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0


class AsyncActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a chunked policy with background inference.

    The first call blocks until an action chunk is available. After that, calls
    return the next queued action immediately. When the queue gets short, a
    worker thread starts inference for the next chunk using the latest
    observation seen by ``infer``.

    If the control loop drains the queue before the worker returns, the broker
    repeats the last action by default. For position-target controllers this is
    a hold command, which is safer than blocking the control loop.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        *,
        prefetch_at: Optional[int] = None,
        first_action_timeout_s: Optional[float] = None,
        max_stale_steps: Optional[int] = None,
    ):
        if action_horizon <= 0:
            raise ValueError("action_horizon must be > 0")
        if prefetch_at is None:
            prefetch_at = action_horizon - 1
        if prefetch_at < 0:
            raise ValueError("prefetch_at must be >= 0")

        self._policy = policy
        self._action_horizon = action_horizon
        self._prefetch_at = min(prefetch_at, action_horizon)
        self._first_action_timeout_s = first_action_timeout_s
        self._max_stale_steps = max_stale_steps

        self._queue: Deque[Dict] = deque()
        self._last_action: Optional[Dict] = None
        self._latest_obs: Optional[Dict] = None
        self._stale_steps = 0
        self._infer_count = 0
        self._underrun_count = 0

        self._generation = 0
        self._inflight = False
        self._last_error: Optional[BaseException] = None
        self._closed = False

        self._lock = threading.Lock()
        self._chunk_ready = threading.Condition(self._lock)
        self._request_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, name="openpi-async-action-broker", daemon=True)
        self._worker.start()

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        with self._chunk_ready:
            if self._closed:
                raise RuntimeError("AsyncActionChunkBroker is closed")

            self._latest_obs = copy.deepcopy(obs)
            self._request_if_needed_locked()

            action = self._pop_action_locked()
            if action is not None:
                return action

            if self._last_action is None:
                action = self._wait_for_action_locked(self._first_action_timeout_s)
                if action is None:
                    raise TimeoutError("Timed out waiting for the first action chunk")
                return action

            if self._max_stale_steps is not None and self._stale_steps >= self._max_stale_steps:
                action = self._wait_for_action_locked(None)
                if action is not None:
                    return action

            self._underrun_count += 1
            self._stale_steps += 1
            return _copy_tree(self._last_action)

    @override
    def reset(self) -> None:
        with self._chunk_ready:
            self._generation += 1
            self._queue.clear()
            self._latest_obs = None
            self._last_action = None
            self._stale_steps = 0
            self._last_error = None
            self._request_event.clear()
            self._chunk_ready.notify_all()
        self._policy.reset()

    def close(self) -> None:
        with self._chunk_ready:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            self._request_event.set()
            self._chunk_ready.notify_all()
        self._worker.join(timeout=1.0)

    def stats(self) -> Dict:
        with self._lock:
            return {
                "queue_size": len(self._queue),
                "inflight": self._inflight,
                "infer_count": self._infer_count,
                "underrun_count": self._underrun_count,
                "stale_steps": self._stale_steps,
            }

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._request_event.wait()
            self._request_event.clear()
            if self._stop_event.is_set():
                return

            with self._chunk_ready:
                if self._latest_obs is None or self._inflight:
                    continue
                obs = copy.deepcopy(self._latest_obs)
                generation = self._generation
                self._inflight = True

            try:
                results = self._policy.infer(obs)
                actions = _split_action_chunk(results, self._action_horizon)
            except BaseException as exc:  # noqa: BLE001
                with self._chunk_ready:
                    self._last_error = exc
                    self._inflight = False
                    self._chunk_ready.notify_all()
                continue

            with self._chunk_ready:
                if generation == self._generation and not self._closed:
                    self._queue.extend(actions)
                    self._infer_count += 1
                    self._last_error = None
                self._inflight = False
                self._chunk_ready.notify_all()

                if len(self._queue) <= self._prefetch_at and self._latest_obs is not None:
                    self._request_if_needed_locked()

    def _pop_action_locked(self) -> Optional[Dict]:
        if not self._queue:
            return None

        action = self._queue.popleft()
        self._last_action = _copy_tree(action)
        self._stale_steps = 0
        self._request_if_needed_locked()
        return action

    def _wait_for_action_locked(self, timeout_s: Optional[float]) -> Optional[Dict]:
        if self._last_error is not None:
            raise RuntimeError("Async policy inference failed") from self._last_error

        self._chunk_ready.wait_for(lambda: bool(self._queue) or self._last_error is not None, timeout=timeout_s)

        if self._last_error is not None:
            raise RuntimeError("Async policy inference failed") from self._last_error

        return self._pop_action_locked()

    def _request_if_needed_locked(self) -> None:
        if self._latest_obs is None or self._inflight or self._closed:
            return
        if len(self._queue) <= self._prefetch_at:
            self._request_event.set()


def _copy_tree(data: Dict) -> Dict:
    def copier(x):
        if isinstance(x, np.ndarray):
            return np.array(x, copy=True)
        return copy.deepcopy(x)

    return tree.map_structure(copier, data)


def _split_action_chunk(results: Dict, action_horizon: int) -> List[Dict]:
    chunks = []
    for step in range(action_horizon):
        chunks.append(tree.map_structure(lambda x: _slice_value(x, step, action_horizon), results))
    return chunks


def _slice_value(value, step: int, action_horizon: int):
    if not isinstance(value, np.ndarray):
        return value
    if value.ndim == 0:
        return value
    if value.shape[0] < action_horizon:
        raise ValueError(
            f"Cannot split action chunk with horizon {action_horizon}: array first dimension is {value.shape[0]}"
        )
    return value[step, ...]
