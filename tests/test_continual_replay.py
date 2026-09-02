import numpy as np
import pytest

from redelex.continual.replay import ReservoirBuffer, herding_select


# --- ReservoirBuffer --------------------------------------------------------


def test_buffer_fills_up_to_capacity():
    buf = ReservoirBuffer(capacity=3, seed=0)
    buf.add([1, 2])
    assert len(buf) == 2
    buf.add([3, 4, 5])
    assert len(buf) == 3
    assert buf.seen == 5


def test_buffer_never_exceeds_capacity():
    buf = ReservoirBuffer(capacity=10, seed=0)
    for _ in range(50):
        buf.add(list(range(20)))
    assert len(buf) == 10
    assert buf.seen == 1000


def test_buffer_keeps_everything_when_under_capacity():
    buf = ReservoirBuffer(capacity=100, seed=0)
    buf.add([5, 6, 7], timestamps=[1, 2, 3], targets=[0.5, 0.6, 0.7])
    np.testing.assert_array_equal(buf.node_ids, [5, 6, 7])
    np.testing.assert_array_equal(buf.timestamps, [1, 2, 3])
    np.testing.assert_allclose(buf.targets, [0.5, 0.6, 0.7])


def test_buffer_sampling_is_approximately_uniform_over_the_stream():
    # The property that matters: old items survive a growing stream. With
    # capacity 50 over 1000 items, each decile should appear ~5 times.
    counts = np.zeros(10)
    trials = 60
    for seed in range(trials):
        buf = ReservoirBuffer(capacity=50, seed=seed)
        buf.add(list(range(1000)))
        deciles = buf.node_ids // 100
        counts += np.bincount(deciles, minlength=10)
    share = counts / counts.sum()
    # each decile should hold ~10% of retained items
    assert np.all(np.abs(share - 0.1) < 0.03), share


def test_buffer_retains_early_items_unlike_recency_sampling():
    buf = ReservoirBuffer(capacity=20, seed=3)
    buf.add(list(range(500)))
    # a recency-biased buffer would hold only ids >= 480
    assert buf.node_ids.min() < 480


def test_buffer_is_deterministic_for_a_seed():
    a = ReservoirBuffer(capacity=7, seed=42)
    b = ReservoirBuffer(capacity=7, seed=42)
    a.add(list(range(100)))
    b.add(list(range(100)))
    np.testing.assert_array_equal(a.node_ids, b.node_ids)


def test_buffer_differs_across_seeds():
    a = ReservoirBuffer(capacity=7, seed=1)
    b = ReservoirBuffer(capacity=7, seed=2)
    a.add(list(range(100)))
    b.add(list(range(100)))
    assert not np.array_equal(a.node_ids, b.node_ids)


def test_buffer_round_trips_through_state_dict():
    buf = ReservoirBuffer(capacity=5, seed=11)
    buf.add(list(range(40)), timestamps=list(range(40)), targets=[0.1] * 40)
    restored = ReservoirBuffer.from_state_dict(buf.state_dict())

    np.testing.assert_array_equal(restored.node_ids, buf.node_ids)
    np.testing.assert_array_equal(restored.timestamps, buf.timestamps)
    np.testing.assert_allclose(restored.targets, buf.targets)
    assert restored.seen == buf.seen
    assert restored.capacity == buf.capacity


def test_restored_buffer_continues_the_same_stream():
    # An episode boundary must not restart the sampling: a restored buffer has
    # to make the same decisions the original would have.
    original = ReservoirBuffer(capacity=6, seed=5)
    original.add(list(range(50)))
    restored = ReservoirBuffer.from_state_dict(original.state_dict())

    original.add(list(range(50, 100)))
    restored.add(list(range(50, 100)))
    np.testing.assert_array_equal(restored.node_ids, original.node_ids)


def test_buffer_add_is_a_noop_for_empty_input():
    buf = ReservoirBuffer(capacity=4, seed=0)
    buf.add([])
    assert len(buf) == 0 and buf.seen == 0


def test_buffer_rejects_non_positive_capacity():
    with pytest.raises(ValueError, match="capacity"):
        ReservoirBuffer(capacity=0)


def test_buffer_rejects_mismatched_lengths():
    buf = ReservoirBuffer(capacity=4, seed=0)
    with pytest.raises(ValueError, match="same length"):
        buf.add([1, 2, 3], timestamps=[1, 2])


# --- herding_select ---------------------------------------------------------


def test_herding_returns_requested_count():
    rng = np.random.default_rng(0)
    assert herding_select(rng.normal(size=(50, 4)), k=8).shape == (8,)


def test_herding_clips_k_to_population():
    assert herding_select(np.zeros((3, 2)), k=99).shape == (3,)


def test_herding_selects_distinct_items():
    rng = np.random.default_rng(1)
    picked = herding_select(rng.normal(size=(30, 3)), k=10)
    assert len(set(picked.tolist())) == 10


def test_herding_mean_is_closer_to_true_mean_than_random():
    # the whole point of herding: better mean coverage at equal budget
    rng = np.random.default_rng(7)
    feats = rng.normal(size=(200, 8))
    true_mean = feats.mean(axis=0)
    k = 20

    herded = np.linalg.norm(feats[herding_select(feats, k=k)].mean(axis=0) - true_mean)
    random_errors = [
        np.linalg.norm(feats[rng.choice(200, k, replace=False)].mean(axis=0) - true_mean)
        for _ in range(50)
    ]
    assert herded < np.median(random_errors)


def test_herding_picks_the_central_cluster_first():
    # one tight cluster at the mean, a few far outliers: the first pick should
    # come from the cluster, since it moves the running mean closest to target
    feats = np.vstack([np.zeros((10, 2)), np.full((2, 2), 50.0)])
    first = herding_select(feats, k=1)[0]
    assert first < 10


def test_herding_can_return_node_ids():
    feats = np.arange(12, dtype=float).reshape(6, 2)
    ids = np.array([100, 101, 102, 103, 104, 105])
    picked = herding_select(feats, k=3, node_ids=ids)
    assert set(picked.tolist()).issubset(set(ids.tolist()))


def test_herding_rejects_bad_shape():
    with pytest.raises(ValueError, match="2-D"):
        herding_select(np.zeros(5), k=2)


def test_herding_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k"):
        herding_select(np.zeros((4, 2)), k=0)
