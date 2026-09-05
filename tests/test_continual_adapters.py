import pytest
import torch
from torch import nn

from redelex.continual.adapters import (
    AdapterStack,
    BottleneckAdapter,
    freeze_module,
    parameter_counts,
)

CHANNELS = 8
RANK = 3
# down: rank x channels + rank, up: channels x rank + channels
PARAMS_PER_ADAPTER = 2 * CHANNELS * RANK + RANK + CHANNELS


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(4, CHANNELS)


def _randomize(adapter):
    """Move an adapter off its identity initialisation, as training would."""
    with torch.no_grad():
        for param in adapter.parameters():
            param.copy_(torch.randn_like(param))


# --- BottleneckAdapter ------------------------------------------------------


def test_fresh_adapter_is_exactly_identity(x):
    # The property the whole method rests on: inserting an adapter must not
    # change the model's output by even a rounding error.
    adapter = BottleneckAdapter(CHANNELS, rank=RANK)
    assert torch.equal(adapter(x), x)


def test_fresh_adapter_is_identity_with_dropout_while_training(x):
    adapter = BottleneckAdapter(CHANNELS, rank=RANK, dropout=0.5)
    adapter.train()
    assert torch.equal(adapter(x), x)


def test_reset_parameters_restores_identity(x):
    adapter = BottleneckAdapter(CHANNELS, rank=RANK)
    _randomize(adapter)
    assert not torch.equal(adapter(x), x)
    adapter.reset_parameters()
    assert torch.equal(adapter(x), x)


def test_adapter_is_not_identity_once_trained(x):
    adapter = BottleneckAdapter(CHANNELS, rank=RANK)
    _randomize(adapter)
    assert not torch.allclose(adapter(x), x)


def test_adapter_preserves_shape_for_any_leading_dimensions():
    adapter = BottleneckAdapter(CHANNELS, rank=RANK)
    for shape in [(CHANNELS,), (5, CHANNELS), (2, 3, CHANNELS)]:
        assert adapter(torch.randn(*shape)).shape == shape


def test_adapter_parameter_count_is_small_and_exact():
    adapter = BottleneckAdapter(CHANNELS, rank=RANK)
    assert parameter_counts(adapter)["total"] == PARAMS_PER_ADAPTER
    # a bottleneck must cost far less than a dense channels x channels layer
    assert PARAMS_PER_ADAPTER < CHANNELS * CHANNELS * 2


def test_adapter_rejects_last_dimension_mismatch():
    adapter = BottleneckAdapter(CHANNELS, rank=RANK)
    with pytest.raises(ValueError, match="last dimension"):
        adapter(torch.randn(4, CHANNELS + 1))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"channels": 0}, "channels"),
        ({"channels": CHANNELS, "rank": 0}, "rank"),
        ({"channels": CHANNELS, "dropout": 1.0}, "dropout"),
        ({"channels": CHANNELS, "dropout": -0.1}, "dropout"),
    ],
)
def test_adapter_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BottleneckAdapter(**kwargs)


# --- AdapterStack -----------------------------------------------------------


def test_empty_stack_is_identity(x):
    stack = AdapterStack(CHANNELS, rank=RANK)
    assert stack.n_adapters == 0
    assert torch.equal(stack(x), x)


def test_stack_stays_identity_when_an_adapter_is_added(x):
    stack = AdapterStack(CHANNELS, rank=RANK)
    _randomize(stack.add_adapter())
    trained = stack(x)
    stack.add_adapter()
    # the episode boundary must not move the function it inherited
    assert torch.equal(stack(x), trained)


def test_stack_applies_adapters_in_order(x):
    stack = AdapterStack(CHANNELS, rank=RANK)
    first, second = stack.add_adapter(), stack.add_adapter()
    _randomize(first)
    _randomize(second)
    torch.testing.assert_close(stack(x), second(first(x)))


def test_stack_preserves_shape(x):
    stack = AdapterStack(CHANNELS, rank=RANK)
    for _ in range(3):
        _randomize(stack.add_adapter())
    assert stack(x).shape == x.shape


def test_only_the_newest_adapter_is_trainable():
    stack = AdapterStack(CHANNELS, rank=RANK)
    for _ in range(3):
        stack.add_adapter()
    for adapter in stack.adapters[:-1]:
        assert all(not p.requires_grad for p in adapter.parameters())
    assert all(p.requires_grad for p in stack.adapters[-1].parameters())


def test_trainable_parameters_are_exactly_the_newest_adapters():
    stack = AdapterStack(CHANNELS, rank=RANK)
    stack.add_adapter()
    newest = stack.add_adapter()
    trainable = list(stack.trainable_parameters())
    assert len(trainable) == len(list(newest.parameters()))
    assert all(any(p is q for q in newest.parameters()) for p in trainable)


def test_trainable_parameters_is_empty_before_the_first_episode():
    assert list(AdapterStack(CHANNELS, rank=RANK).trainable_parameters()) == []


def test_stack_grows_by_one_adapters_worth_of_parameters():
    stack = AdapterStack(CHANNELS, rank=RANK)
    for i in range(1, 4):
        stack.add_adapter()
        counts = parameter_counts(stack)
        assert counts["total"] == i * PARAMS_PER_ADAPTER
        # capacity accumulates, but only one adapter is ever being optimised
        assert counts["trainable"] == PARAMS_PER_ADAPTER


def test_add_adapter_accepts_a_per_episode_rank():
    stack = AdapterStack(CHANNELS, rank=RANK)
    wide = stack.add_adapter(rank=RANK * 2)
    assert wide.rank == RANK * 2
    assert parameter_counts(stack)["total"] == 2 * CHANNELS * RANK * 2 + RANK * 2 + CHANNELS


def test_gradients_reach_only_the_newest_adapter(x):
    stack = AdapterStack(CHANNELS, rank=RANK)
    old = stack.add_adapter()
    _randomize(old)
    freeze_module(old, freeze=True)  # add_adapter would do this; be explicit
    new = stack.add_adapter()

    stack(x).sum().backward()

    assert all(p.grad is None for p in old.parameters())
    assert all(p.grad is not None for p in new.parameters())
    # the up-projection is where learning starts, since it is the zeroed side
    assert torch.any(new.up.weight.grad != 0)


def test_training_the_newest_adapter_leaves_older_ones_untouched(x):
    stack = AdapterStack(CHANNELS, rank=RANK)
    old = stack.add_adapter()
    _randomize(old)
    stack.add_adapter()
    before = [p.detach().clone() for p in old.parameters()]

    optimizer = torch.optim.SGD(stack.trainable_parameters(), lr=0.1)
    for _ in range(5):
        optimizer.zero_grad()
        stack(x).pow(2).sum().backward()
        optimizer.step()

    # zero forgetting by construction: the old episode's weights never move
    for param, original in zip(old.parameters(), before):
        assert torch.equal(param, original)


def test_a_new_adapter_can_actually_learn(x):
    target = torch.zeros_like(x)
    stack = AdapterStack(CHANNELS, rank=RANK)
    stack.add_adapter()
    optimizer = torch.optim.SGD(stack.trainable_parameters(), lr=0.05)

    start = nn.functional.mse_loss(stack(x), target).item()
    for _ in range(50):
        optimizer.zero_grad()
        nn.functional.mse_loss(stack(x), target).backward()
        optimizer.step()

    assert nn.functional.mse_loss(stack(x), target).item() < start


# --- state_dict round-trip --------------------------------------------------


def test_state_dict_round_trips_across_an_episode_boundary(x):
    stack = AdapterStack(CHANNELS, rank=RANK)
    for _ in range(3):
        _randomize(stack.add_adapter())

    restored = AdapterStack(CHANNELS, rank=RANK)
    restored.load_state_dict(stack.state_dict())

    assert restored.n_adapters == 3
    assert torch.equal(restored(x), stack(x))


def test_loading_restores_the_freeze_pattern():
    stack = AdapterStack(CHANNELS, rank=RANK)
    for _ in range(3):
        stack.add_adapter()

    restored = AdapterStack(CHANNELS, rank=RANK)
    restored.load_state_dict(stack.state_dict())

    assert all(not p.requires_grad for p in restored.adapters[0].parameters())
    assert all(p.requires_grad for p in restored.adapters[-1].parameters())


def test_loading_an_earlier_checkpoint_shrinks_the_stack(x):
    early = AdapterStack(CHANNELS, rank=RANK)
    _randomize(early.add_adapter())

    late = AdapterStack(CHANNELS, rank=RANK)
    for _ in range(3):
        _randomize(late.add_adapter())
    late.load_state_dict(early.state_dict())

    assert late.n_adapters == 1
    assert torch.equal(late(x), early(x))


def test_state_dict_round_trips_with_per_episode_ranks(x):
    stack = AdapterStack(CHANNELS, rank=RANK)
    _randomize(stack.add_adapter(rank=2))
    _randomize(stack.add_adapter(rank=5))

    restored = AdapterStack(CHANNELS, rank=RANK)
    restored.load_state_dict(stack.state_dict())

    assert [a.rank for a in restored.adapters] == [2, 5]
    assert torch.equal(restored(x), stack(x))


def test_state_dict_round_trips_when_nested_in_a_model(x):
    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Linear(CHANNELS, CHANNELS)
            self.adapters = AdapterStack(CHANNELS, rank=RANK)

        def forward(self, inputs):
            return self.adapters(self.body(inputs))

    model = Wrapped()
    for _ in range(2):
        _randomize(model.adapters.add_adapter())

    restored = Wrapped()
    restored.load_state_dict(model.state_dict())

    assert restored.adapters.n_adapters == 2
    assert torch.equal(restored(x), model(x))


def test_loading_a_stack_checkpoint_is_strict_about_other_keys():
    stack = AdapterStack(CHANNELS, rank=RANK)
    stack.add_adapter()
    state = stack.state_dict()
    state["adapters.0.mystery"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="[Uu]nexpected"):
        AdapterStack(CHANNELS, rank=RANK).load_state_dict(state)


# --- freeze_module ----------------------------------------------------------


def test_freeze_module_reports_how_many_tensors_changed():
    model = nn.Linear(4, 2)  # weight + bias
    assert freeze_module(model, freeze=True) == 2
    assert all(not p.requires_grad for p in model.parameters())


def test_freeze_module_is_a_no_op_when_already_in_that_state():
    model = nn.Linear(4, 2)
    freeze_module(model, freeze=True)
    assert freeze_module(model, freeze=True) == 0


def test_freeze_module_is_reversible():
    model = nn.Linear(4, 2)
    freeze_module(model, freeze=True)
    assert freeze_module(model, freeze=False) == 2
    assert all(p.requires_grad for p in model.parameters())


def test_freeze_module_counts_only_the_tensors_that_flipped():
    model = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 2))
    freeze_module(model[0], freeze=True)
    # two of the four tensors are already frozen
    assert freeze_module(model, freeze=True) == 2


def test_freezing_one_half_of_a_model_blocks_its_gradients():
    encoder, head = nn.Linear(4, 4), nn.Linear(4, 1)
    model = nn.Sequential(encoder, head)
    freeze_module(encoder, freeze=True)

    model(torch.randn(3, 4)).sum().backward()

    assert all(p.grad is None for p in encoder.parameters())
    assert all(p.grad is not None for p in head.parameters())


# --- parameter_counts -------------------------------------------------------


def test_parameter_counts_splits_total_into_trainable_and_frozen():
    model = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 3))
    counts = parameter_counts(model)
    assert counts["total"] == (4 * 2 + 2) + (2 * 3 + 3)
    assert counts["trainable"] == counts["total"]
    assert counts["frozen"] == 0


def test_parameter_counts_follows_freezing():
    model = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 3))
    freeze_module(model[0], freeze=True)
    counts = parameter_counts(model)
    assert counts["frozen"] == 4 * 2 + 2
    assert counts["trainable"] + counts["frozen"] == counts["total"]


def test_parameter_counts_of_an_empty_module_is_zero():
    counts = parameter_counts(nn.ReLU())
    assert counts == {"total": 0, "trainable": 0, "frozen": 0}


# --- regressions found by adversarial verification --------------------------


class _WrappedBody(nn.Module):
    """A backbone with an adapter stack hanging off it, as the experiment builds."""

    def __init__(self, channels=8, rank=3):
        super().__init__()
        self.body = nn.Linear(channels, channels)
        self.adapters = AdapterStack(channels, rank)


def test_partial_load_does_not_destroy_the_adapter_stack():
    # The freeze_extend chain loads a checkpoint with strict=False, because the
    # stack grows by one adapter per episode and no checkpoint ever has exactly
    # the model's keys. Treating "no adapter keys" as "zero adapters" silently
    # discarded every accumulated adapter -- and with it the zero-forgetting
    # guarantee -- reporting no missing keys and no warning.
    model = _WrappedBody()
    model.adapters.add_adapter()
    model.adapters.add_adapter()

    result = model.load_state_dict(
        {"body.weight": torch.randn(8, 8), "body.bias": torch.randn(8)}, strict=False
    )

    assert model.adapters.n_adapters == 2
    assert parameter_counts(model.adapters)["total"] > 0
    assert result.missing_keys  # the adapter keys are genuinely absent, and reported


def test_full_load_still_resizes_the_stack_to_match():
    # The counterpart: when the incoming dict DOES describe adapters, the stack
    # must still resize, or a resumed chain would silently run the wrong depth.
    src = AdapterStack(8, 3)
    src.add_adapter()
    src.add_adapter()
    src.add_adapter()

    dst = AdapterStack(8, 3)
    dst.load_state_dict(src.state_dict())
    assert dst.n_adapters == 3


def test_same_rank_load_restores_the_freeze_invariant():
    # _rebuild restored the invariant, but it only runs when ranks changed, so a
    # same-shaped load inherited whatever requires_grad state the destination had.
    src = AdapterStack(8, 3)
    src.add_adapter()
    src.add_adapter()

    dst = AdapterStack(8, 3)
    dst.add_adapter()
    dst.add_adapter()
    freeze_module(dst, freeze=False)  # everything trainable, the wrong state

    dst.load_state_dict(src.state_dict())

    assert not any(p.requires_grad for p in dst.adapters[0].parameters())
    assert all(p.requires_grad for p in dst.adapters[1].parameters())


def test_failed_add_adapter_leaves_the_stack_trainable():
    # add_adapter froze the existing adapters before constructing the new one, so
    # a constructor failure left nothing trainable and the next backward pass died
    # with "element 0 of tensors does not require grad" far from the real cause.
    stack = AdapterStack(8, 3)
    stack.add_adapter()
    before = parameter_counts(stack)["trainable"]

    with pytest.raises(ValueError):
        stack.add_adapter(rank=0)

    assert parameter_counts(stack)["trainable"] == before
    assert before > 0
    stack(torch.randn(2, 8)).pow(2).sum().backward()  # must not raise


def test_dropout_is_actually_wired_into_the_forward_pass():
    # Deleting the dropout call from BottleneckAdapter.forward survived the whole
    # suite: the identity tests pass either way, because the zero-initialised up
    # projection masks it.
    adapter = BottleneckAdapter(channels=16, rank=8, dropout=0.9)
    with torch.no_grad():  # break the identity so dropout can show through
        adapter.up.weight.normal_()
        adapter.up.bias.normal_()
    x = torch.randn(64, 16)

    adapter.train()
    torch.manual_seed(0)
    a = adapter(x)
    torch.manual_seed(1)
    b = adapter(x)
    assert not torch.allclose(a, b), "train mode must be stochastic when dropout > 0"

    adapter.eval()
    assert torch.allclose(adapter(x), adapter(x)), "eval mode must be deterministic"
