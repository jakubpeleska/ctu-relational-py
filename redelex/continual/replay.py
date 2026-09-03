r"""Bounded exemplar buffers for replay-based continual learning.

The existing ``ft_upsample`` regime re-samples the *entire* history every
episode, which is joint training with a reweighted loader rather than replay. A
replay method keeps a **fixed-size** buffer that must be carried from episode to
episode and refreshed as new data arrives, so buffer size becomes the knob that
trades memory against forgetting.

Two selection strategies are provided:

* :class:`ReservoirBuffer` -- uniform sampling over the whole stream in one pass
  (Vitter's Algorithm R). Cheap, needs no model, and is the standard replay
  baseline.
* :func:`herding_select` -- iCaRL-style greedy selection of exemplars whose
  running mean best tracks the mean embedding. Needs embeddings, but picks a far
  more representative subset at the same budget.

Buffers are plain arrays of entity ids, timestamps and targets, so they
serialise with :meth:`~ReservoirBuffer.state_dict` and travel alongside the
weights that already move between episodes.
"""

from typing import Optional

import numpy as np

__all__ = ["ReservoirBuffer", "herding_select"]


class ReservoirBuffer:
    r"""Fixed-capacity uniform sample of everything added so far.

    After :meth:`add` has seen ``n`` items, every item is present with
    probability ``min(1, capacity / n)``, regardless of when it arrived. That
    uniformity is the point: it keeps old episodes represented as history grows,
    which proportional sampling over a growing history does not.

    Args:
        capacity: Maximum number of exemplars retained.
        seed: Seed for the internal RNG, so a resumed run rebuilds the same
            buffer.

    Example:
        >>> buf = ReservoirBuffer(capacity=2, seed=0)
        >>> buf.add([1, 2, 3], timestamps=[10, 20, 30], targets=[0.0, 1.0, 0.0])
        >>> len(buf)
        2
        >>> buf.seen
        3
    """

    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError(f"`capacity` must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self.seen = 0
        self.node_ids = np.empty(0, dtype=np.int64)
        self.timestamps = np.empty(0, dtype=np.int64)
        self.targets = np.empty(0, dtype=np.float64)
        # Model output recorded when the exemplar was stored. DER++ distils
        # against these rather than against a teacher's fresh predictions, which
        # is what lets it work without keeping a copy of the previous model.
        self.logits = np.empty(0, dtype=np.float64)

    def __len__(self) -> int:
        return int(self.node_ids.shape[0])

    def add(self, node_ids, timestamps=None, targets=None, logits=None) -> None:
        """Offer a batch of items to the buffer, retaining a uniform sample."""
        ids = np.asarray(node_ids, dtype=np.int64).ravel()
        if ids.size == 0:
            return
        times = (
            np.zeros(ids.shape, dtype=np.int64)
            if timestamps is None
            else np.asarray(timestamps, dtype=np.int64).ravel()
        )
        vals = (
            np.zeros(ids.shape, dtype=np.float64)
            if targets is None
            else np.asarray(targets, dtype=np.float64).ravel()
        )
        outs = (
            np.zeros(ids.shape, dtype=np.float64)
            if logits is None
            else np.asarray(logits, dtype=np.float64).ravel()
        )
        if times.shape != ids.shape or vals.shape != ids.shape or outs.shape != ids.shape:
            raise ValueError(
                f"node_ids {ids.shape}, timestamps {times.shape}, targets "
                f"{vals.shape} and logits {outs.shape} must have the same length"
            )

        for i in range(ids.shape[0]):
            self.seen += 1
            if len(self) < self.capacity:
                self.node_ids = np.append(self.node_ids, ids[i])
                self.timestamps = np.append(self.timestamps, times[i])
                self.targets = np.append(self.targets, vals[i])
                self.logits = np.append(self.logits, outs[i])
                continue
            # Replace a uniformly chosen slot with probability capacity / seen.
            j = int(self._rng.integers(0, self.seen))
            if j < self.capacity:
                self.node_ids[j] = ids[i]
                self.timestamps[j] = times[i]
                self.targets[j] = vals[i]
                self.logits[j] = outs[i]

    def state_dict(self) -> dict:
        """Serialisable state, to persist the buffer between episodes."""
        return {
            "capacity": self.capacity,
            "seed": self.seed,
            "seen": self.seen,
            "node_ids": self.node_ids.copy(),
            "timestamps": self.timestamps.copy(),
            "targets": self.targets.copy(),
            "logits": self.logits.copy(),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state produced by :meth:`state_dict`."""
        self.capacity = int(state["capacity"])
        self.seed = int(state["seed"])
        self.seen = int(state["seen"])
        self.node_ids = np.asarray(state["node_ids"], dtype=np.int64).copy()
        self.timestamps = np.asarray(state["timestamps"], dtype=np.int64).copy()
        self.targets = np.asarray(state["targets"], dtype=np.float64).copy()
        self.logits = np.asarray(
            state.get("logits", np.zeros_like(self.targets)), dtype=np.float64
        ).copy()
        if state.get("rng_state") is not None:
            self._rng.bit_generator.state = state["rng_state"]

    @classmethod
    def from_state_dict(cls, state: dict) -> "ReservoirBuffer":
        buf = cls(capacity=int(state["capacity"]), seed=int(state["seed"]))
        buf.load_state_dict(state)
        return buf


def herding_select(
    embeddings,
    k: int,
    node_ids=None,
) -> np.ndarray:
    r"""Greedily pick ``k`` exemplars whose running mean tracks the overall mean.

    The iCaRL construction (Rebuffi et al., 2017): repeatedly take the item that
    brings the mean of the chosen set closest to the mean of the full set. At
    equal budget this covers the distribution far better than uniform sampling,
    at the cost of needing embeddings.

    Args:
        embeddings: Array of shape ``(n, d)``.
        k: Number of exemplars to select. Clipped to ``n``.
        node_ids: Optional ids to return instead of positional indices.

    Returns:
        Selected indices (or ``node_ids`` values), in selection order.
    """
    feats = np.asarray(embeddings, dtype=np.float64)
    if feats.ndim != 2:
        raise ValueError(f"`embeddings` must be 2-D, got shape {feats.shape}")
    n = feats.shape[0]
    if k <= 0:
        raise ValueError(f"`k` must be positive, got {k}")
    k = min(int(k), n)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    target_mean = feats.mean(axis=0)
    chosen: list[int] = []
    # Running sum of the selected embeddings, so each step is a single argmin.
    running = np.zeros(feats.shape[1], dtype=np.float64)
    available = np.ones(n, dtype=bool)

    for step in range(k):
        # Candidate mean if each remaining item were added next.
        candidate_means = (running + feats) / (step + 1)
        distances = np.linalg.norm(candidate_means - target_mean, axis=1)
        distances[~available] = np.inf
        pick = int(np.argmin(distances))
        chosen.append(pick)
        available[pick] = False
        running = running + feats[pick]

    idx = np.asarray(chosen, dtype=np.int64)
    if node_ids is None:
        return idx
    return np.asarray(node_ids)[idx]
