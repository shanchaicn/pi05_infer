import threading
import time

import numpy as np

from openpi_client import action_chunk_broker
from openpi_client import base_policy


class _FakeChunkPolicy(base_policy.BasePolicy):
    def __init__(self, *, block_after_first: bool = False):
        self.calls = 0
        self.block_after_first = block_after_first
        self.release = threading.Event()

    def infer(self, obs):  # noqa: ARG002
        self.calls += 1
        if self.block_after_first and self.calls > 1:
            self.release.wait(timeout=1.0)

        base = 10 * self.calls
        return {"actions": np.arange(base, base + 6).reshape(3, 2)}


def test_async_broker_returns_chunk_actions_in_order():
    policy = _FakeChunkPolicy()
    broker = action_chunk_broker.AsyncActionChunkBroker(policy, action_horizon=3, prefetch_at=0)

    try:
        np.testing.assert_array_equal(broker.infer({"step": 0})["actions"], np.array([10, 11]))
        np.testing.assert_array_equal(broker.infer({"step": 1})["actions"], np.array([12, 13]))
        np.testing.assert_array_equal(broker.infer({"step": 2})["actions"], np.array([14, 15]))
    finally:
        broker.close()


def test_async_broker_holds_last_action_on_underrun():
    policy = _FakeChunkPolicy(block_after_first=True)
    broker = action_chunk_broker.AsyncActionChunkBroker(policy, action_horizon=3, prefetch_at=1)

    try:
        np.testing.assert_array_equal(broker.infer({"step": 0})["actions"], np.array([10, 11]))
        np.testing.assert_array_equal(broker.infer({"step": 1})["actions"], np.array([12, 13]))
        np.testing.assert_array_equal(broker.infer({"step": 2})["actions"], np.array([14, 15]))

        start = time.monotonic()
        held_action = broker.infer({"step": 3})["actions"]
        elapsed = time.monotonic() - start

        np.testing.assert_array_equal(held_action, np.array([14, 15]))
        assert elapsed < 0.1
        assert broker.stats()["underrun_count"] == 1
    finally:
        policy.release.set()
        broker.close()
