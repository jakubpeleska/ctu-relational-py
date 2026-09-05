r"""Parameter isolation: freeze what was learned, extend with a small adapter.

The regimes in this benchmark all keep training the *same* weights, so every
episode overwrites what the previous one learned. Parameter isolation removes
that failure mode by construction: the parameters of past episodes are frozen
and never touched again, and each new episode gets a small trainable module of
its own. Forgetting is then exactly zero -- the only thing at stake is whether
the added capacity is enough to fit the new episode.

The added module is a **bottleneck adapter** (Houlsby et al., 2019): project
``channels`` down to ``rank``, apply a nonlinearity, project back, and add the
result to the input. At ``rank << channels`` this costs a few thousand
parameters per episode instead of a full model, which matters over the 52-episode
chains here.

The up-projection starts at zero, so a freshly inserted adapter is an *exact*
identity. Without that, bolting an adapter onto a model that already scores well
would perturb it before a single gradient step, and the first episode after
insertion would measure the perturbation rather than the method.

:func:`freeze_module` and :func:`parameter_counts` support the same family of
experiments: freezing one half of the network (tabular encoder vs. GNN) to
attribute forgetting to a component, and reporting how much capacity each method
actually adds.
"""

from typing import Dict, Iterator, List, Optional

import torch
from torch import nn

__all__ = ["BottleneckAdapter", "AdapterStack", "freeze_module", "parameter_counts"]


class BottleneckAdapter(nn.Module):
    r"""Residual down-project / nonlinearity / up-project block.

    ``forward(x) = x + W_up(relu(W_down(x)))`` over the last dimension, so the
    output shape always equals the input shape and the block can be inserted
    anywhere a residual connection is valid.

    ``W_up`` is initialised to zero (see :meth:`reset_parameters`), making a new
    adapter an exact identity at insertion.

    Args:
        channels: Width of the representation the adapter wraps.
        rank: Bottleneck width. Keep it well below ``channels``; this is the
            knob that trades added capacity against added parameters.
        dropout: Dropout applied to the bottleneck activations during training.

    Raises:
        ValueError: If ``channels`` or ``rank`` is not positive, or ``dropout``
            is outside ``[0, 1)``.

    Example:
        >>> import torch
        >>> adapter = BottleneckAdapter(channels=8, rank=2)
        >>> x = torch.randn(4, 8)
        >>> torch.equal(adapter(x), x)   # identity until it is trained
        True
    """

    def __init__(self, channels: int, rank: int = 16, dropout: float = 0.0):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"`channels` must be positive, got {channels}")
        if rank <= 0:
            raise ValueError(f"`rank` must be positive, got {rank}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}")

        self.channels = int(channels)
        self.rank = int(rank)

        self.down = nn.Linear(self.channels, self.rank)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(self.rank, self.channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialise, restoring the exact-identity property."""
        self.down.reset_parameters()
        # Zero up-projection => zero residual branch => exact identity. The
        # down-projection stays randomly initialised so the two projections do
        # not start symmetric, which would keep the branch collapsed.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.channels:
            raise ValueError(
                f"expected last dimension {self.channels}, got {tuple(x.shape)}"
            )
        return x + self.up(self.dropout(self.activation(self.down(x))))

    def extra_repr(self) -> str:
        return f"channels={self.channels}, rank={self.rank}"


class AdapterStack(nn.Module):
    r"""Ordered adapters, one per episode, with only the newest one trainable.

    :meth:`add_adapter` is the episode boundary: it freezes everything learned
    so far and opens exactly one new module for training. Since the frozen
    adapters and the identity-initialised new one leave the function unchanged
    at that moment, the model that enters an episode is bit-for-bit the model
    that left the previous one.

    Args:
        channels: Width of the representation being adapted.
        rank: Default bottleneck width for adapters added later.
        dropout: Default dropout for adapters added later.

    Example:
        >>> stack = AdapterStack(channels=8, rank=2)
        >>> _ = stack.add_adapter()          # episode 0
        >>> _ = stack.add_adapter()          # episode 1: episode 0 is now frozen
        >>> stack.n_adapters
        2
        >>> all(not p.requires_grad for p in stack.adapters[0].parameters())
        True
    """

    def __init__(self, channels: int, rank: int = 16, dropout: float = 0.0):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"`channels` must be positive, got {channels}")
        if rank <= 0:
            raise ValueError(f"`rank` must be positive, got {rank}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}")

        self.channels = int(channels)
        self.rank = int(rank)
        self.dropout = float(dropout)
        self.adapters = nn.ModuleList()
        # Non-persistent, so it never enters the state dict: it exists purely so
        # adapters rebuilt while loading land on the device the stack lives on.
        self.register_buffer("_device_probe", torch.empty(0), persistent=False)

    @property
    def n_adapters(self) -> int:
        """Number of adapters currently held."""
        return len(self.adapters)

    def __len__(self) -> int:
        return len(self.adapters)

    def add_adapter(self, rank: Optional[int] = None) -> BottleneckAdapter:
        r"""Freeze every existing adapter and append a fresh trainable one.

        Args:
            rank: Bottleneck width for the new adapter. Defaults to the stack's
                ``rank``, but may vary per episode.

        Returns:
            The new adapter, the only trainable module in the stack.
        """
        # Construct BEFORE freezing. Freezing first would leave the stack with no
        # trainable module if the constructor raises -- the next backward pass then
        # fails with "element 0 of tensors does not require grad", far from the cause.
        adapter = BottleneckAdapter(
            channels=self.channels,
            rank=self.rank if rank is None else rank,
            dropout=self.dropout,
        ).to(device=self._device_probe.device, dtype=self._device_probe.dtype)
        freeze_module(self.adapters, freeze=True)
        self.adapters.append(adapter)
        return adapter

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        r"""Parameters of the newest adapter only -- what the optimiser gets.

        Yields nothing while the stack is empty, so the caller can build an
        optimiser without special-casing episode 0.
        """
        if not self.adapters:
            return iter(())
        return self.adapters[-1].parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for adapter in self.adapters:
            x = adapter(x)
        return x

    def _rebuild(self, ranks: List[int]) -> None:
        """Replace the adapters with fresh ones of the given ranks."""
        self.adapters = nn.ModuleList(
            BottleneckAdapter(
                channels=self.channels, rank=rank, dropout=self.dropout
            ).to(device=self._device_probe.device, dtype=self._device_probe.dtype)
            for rank in ranks
        )
        # Restore the invariant: everything but the newest adapter is frozen.
        freeze_module(self.adapters, freeze=True)
        if self.adapters:
            freeze_module(self.adapters[-1], freeze=False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        # How many adapters a stack holds is data, not architecture: a checkpoint
        # written after episode k carries k adapters while a freshly constructed
        # stack has none. Grow (or shrink) to match before the base loader walks
        # the children, otherwise every adapter key is a missing/unexpected key.
        ranks: List[int] = []
        while True:
            weight = state_dict.get(f"{prefix}adapters.{len(ranks)}.down.weight")
            if weight is None:
                break
            ranks.append(int(weight.shape[0]))

        # A state dict with NO adapter keys is ambiguous: under strict=True it means
        # "this checkpoint had zero adapters", but under strict=False it means "not
        # provided". Resizing to zero on the second reading silently destroys an
        # already-restored stack -- and with it the zero-forgetting guarantee -- with
        # no missing keys and no warning. Only ever resize when the incoming dict
        # actually describes adapters; strict mode still reports the mismatch itself.
        has_adapter_keys = any(
            key.startswith(f"{prefix}adapters.") for key in state_dict
        )
        if has_adapter_keys and ranks != [adapter.rank for adapter in self.adapters]:
            self._rebuild(ranks)

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

        # Restore the invariant on every path. `_rebuild` does it too, but it only
        # runs when the ranks changed -- loading a same-shaped stack would otherwise
        # inherit whatever requires_grad state the destination happened to have.
        if self.adapters:
            freeze_module(self.adapters, freeze=True)
            freeze_module(self.adapters[-1], freeze=False)


def freeze_module(module: nn.Module, freeze: bool = True) -> int:
    r"""Set ``requires_grad`` on every parameter of ``module``.

    Args:
        module: Module to freeze or unfreeze, in place.
        freeze: ``True`` freezes (``requires_grad = False``), ``False`` unfreezes.

    Returns:
        Number of parameter tensors whose ``requires_grad`` actually changed, so
        a caller can log "froze 12 tensors" and notice a no-op mistake such as
        freezing an already frozen encoder.
    """
    target = not freeze
    changed = 0
    for param in module.parameters():
        if param.requires_grad != target:
            param.requires_grad_(target)
            changed += 1
    return changed


def parameter_counts(model: nn.Module) -> Dict[str, int]:
    r"""Element counts of ``model``'s parameters, split by trainability.

    Args:
        model: Any module. Shared parameters are counted once, as
            :meth:`~torch.nn.Module.parameters` deduplicates them.

    Returns:
        Mapping with ``'total'``, ``'trainable'`` and ``'frozen'`` element
        counts, where ``trainable + frozen == total``.
    """
    total = 0
    trainable = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
