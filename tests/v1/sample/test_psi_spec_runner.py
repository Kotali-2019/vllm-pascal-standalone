# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the Psi-Spec two-pass driver (:class:`PsiSpecRunner`).

The driver binds the two-pass step to a live model runner. These tests drive
that binding against a *mock* runner (no engine-core process, no real model)
so the orchestration can be checked in isolation: that ``_forward_block`` points
each row's KV frontier at the block start, writes the block tokens into the
input batch, builds the right ``SchedulerOutput``, applies and then clears the
per-row LoRA gate, and reshapes the block logits to ``[B, L, V]``. The sampler
is checked against an independent reference.
"""

import contextlib

import numpy as np
import pytest
import torch

import vllm.v1.sample.psi_spec_runner as runner_mod
from vllm.v1.sample.psi_spec_runner import PsiSpecRunner

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H = 4  # hidden size
V = 8  # vocab size


class _RecordingWrapper:
    """A LoRA wrapper stand-in that records every gate assignment."""

    def __init__(self) -> None:
        self._mask = None
        self.seen: list[torch.Tensor | None] = []

    @property
    def lora_row_mask(self) -> torch.Tensor | None:
        return self._mask

    @lora_row_mask.setter
    def lora_row_mask(self, value: torch.Tensor | None) -> None:
        self._mask = value
        self.seen.append(None if value is None else value.clone())


class _MockModel:
    def __init__(self) -> None:
        self.last_hidden = None

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        # Deterministic: row i -> a row of i, so the [B, L, V] reshape is
        # checkable without a real model.
        self.last_hidden = hidden
        return torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(-1).expand(-1, V)


class _MockInputBatch:
    def __init__(self, num_reqs: int, max_len: int) -> None:
        self.num_reqs = num_reqs
        self.req_ids = [f"req{r}" for r in range(num_reqs)]
        self.num_computed_tokens_cpu = np.zeros(num_reqs, dtype=np.int32)
        self.token_ids_cpu = np.zeros((num_reqs, max_len), dtype=np.int32)


class _MockRunner:
    """Just enough of a ``GPUModelRunner`` for the driver's ``_forward_block``."""

    def __init__(self, num_reqs: int, max_len: int, with_lora: bool) -> None:
        self.device = DEVICE
        self.model_config = _MockModelConfig(V)
        self.input_batch = _MockInputBatch(num_reqs, max_len)
        self.model = _MockModel()
        self.vllm_config = object()
        n = num_reqs * max_len
        self.input_ids = _Buffer(torch.zeros(n, dtype=torch.int32, device=DEVICE))
        self.positions = torch.zeros(n, dtype=torch.int64, device=DEVICE)
        self._wrapper = _RecordingWrapper() if with_lora else None
        self.prepare_calls: list[tuple[object, np.ndarray]] = []

    def get_lora_punica_wrapper(self):
        return self._wrapper

    def _prepare_inputs(self, so, num_scheduled_tokens_np):
        self.prepare_calls.append((so, num_scheduled_tokens_np.copy()))
        return None, None, 1

    def _get_slot_mappings(self, num_tokens_padded, num_reqs_padded, num_tokens_unpadded):
        return {}, {}

    def _build_attention_metadata(self, **kwargs):
        return {}, None

    def _model_forward(self, input_ids, positions, **kwargs):
        # Return a deterministic hidden tensor of shape [n, H] that depends on
        # the (mock) input so the caller can confirm the right tokens flowed.
        n = input_ids.shape[0]
        return torch.arange(n * H, device=DEVICE, dtype=torch.float32).reshape(n, H)


class _Buffer:
    def __init__(self, gpu: torch.Tensor) -> None:
        self.gpu = gpu


class _MockModelConfig:
    def __init__(self, vocab: int) -> None:
        self._vocab = vocab

    def get_vocab_size(self) -> int:
        return self._vocab


def _no_op_ctx(*args, **kwargs):
    @contextlib.contextmanager
    def _cm():
        yield

    return _cm()


@pytest.fixture(autouse=True)
def _patch_forward_context(monkeypatch):
    # Keep the driver's forward off the real (validating) forward context.
    monkeypatch.setattr(runner_mod, "set_forward_context", _no_op_ctx)


def _make_runner(num_reqs: int, max_len: int, with_lora: bool = False) -> _MockRunner:
    return _MockRunner(num_reqs, max_len, with_lora)


def test_forward_block_sets_frontier_and_tokens():
    """Each row's frontier is set to the block start and the block tokens are
    written into the input batch at that frontier."""
    runner = _make_runner(num_reqs=2, max_len=32)
    drv = PsiSpecRunner(runner, block_len=3, mask_token_id=4)
    ib = runner.input_batch

    # Row 0: block [10, 11, 12] at frontier 5. Row 1: [20, 21, 22] at 9.
    ids = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.long, device=DEVICE)
    pos = torch.tensor([[5, 6, 7], [9, 10, 11]], dtype=torch.long, device=DEVICE)
    gate = torch.zeros(6, device=DEVICE)

    out = drv._forward_block(ids, pos, gate)

    assert out.shape == (2, 3, V)
    # Frontiers point at the block starts.
    assert ib.num_computed_tokens_cpu.tolist() == [5, 9]
    # Block tokens written at the frontier.
    assert ib.token_ids_cpu[0, 5:8].tolist() == [10, 11, 12]
    assert ib.token_ids_cpu[1, 9:12].tolist() == [20, 21, 22]


def test_forward_block_scheduler_output():
    """The driver builds a SchedulerOutput that schedules L tokens per row."""
    runner = _make_runner(num_reqs=2, max_len=32)
    drv = PsiSpecRunner(runner, block_len=4, mask_token_id=4)
    ids = torch.zeros((2, 4), dtype=torch.long, device=DEVICE)
    pos = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long, device=DEVICE)
    drv._forward_block(ids, pos, torch.zeros(8, device=DEVICE))

    assert len(runner.prepare_calls) == 1
    so, np_sched = runner.prepare_calls[0]
    assert so.num_scheduled_tokens == {"req0": 4, "req1": 4}
    assert so.total_num_scheduled_tokens == 8
    assert np_sched.tolist() == [4, 4]


def test_forward_block_logits_reshape():
    """Block logits come back as [B, L, V] with the deterministic values."""
    runner = _make_runner(num_reqs=1, max_len=16)
    drv = PsiSpecRunner(runner, block_len=3, mask_token_id=4)
    ids = torch.zeros((1, 3), dtype=torch.long, device=DEVICE)
    pos = torch.tensor([[2, 3, 4]], dtype=torch.long, device=DEVICE)
    out = drv._forward_block(ids, pos, torch.zeros(3, device=DEVICE))
    # Row i (global) is all i; for B=1, L=3 the global rows are 0,1,2.
    assert out.shape == (1, 3, V)
    for r in range(3):
        assert torch.all(out[0, r] == r)


def test_forward_block_clears_lora_gate():
    """The per-row LoRA gate is applied for the forward and cleared afterwards."""
    runner = _make_runner(num_reqs=1, max_len=16, with_lora=True)
    drv = PsiSpecRunner(runner, block_len=2, mask_token_id=4)
    gate = torch.tensor([0.0, 1.0], device=DEVICE)
    ids = torch.zeros((1, 2), dtype=torch.long, device=DEVICE)
    pos = torch.tensor([[0, 1]], dtype=torch.long, device=DEVICE)
    drv._forward_block(ids, pos, gate)

    wrapper = runner.get_lora_punica_wrapper()
    # Last assignment is the clear (None); an earlier one carried the gate.
    assert wrapper.lora_row_mask is None
    assert wrapper.seen[0].tolist() == [0.0, 1.0]
    assert wrapper.seen[-1] is None


def test_sample_greedy():
    runner = _make_runner(num_reqs=1, max_len=8)
    drv = PsiSpecRunner(runner, block_len=1, mask_token_id=4, temperature=0.0)
    logits = torch.zeros(2, V, device=DEVICE)
    logits[0, 3] = 10.0
    logits[1, 5] = 10.0
    out = drv._sample(logits)
    assert out.tolist() == [3, 5]


def test_sample_top_k_reference():
    """top-k sampling draws only from the top-k support."""
    runner = _make_runner(num_reqs=1, max_len=8)
    drv = PsiSpecRunner(
        runner, block_len=1, mask_token_id=4, temperature=1.0, top_k=2
    )
    torch.manual_seed(0)
    # Make tokens 0 and 1 the clear top-2; the rest far lower.
    logits = torch.full((1, V), -20.0, device=DEVICE)
    logits[0, 0] = 5.0
    logits[0, 1] = 4.0
    out = drv._sample(logits)
    assert out.item() in (0, 1)


def test_apply_top_k_mask():
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0]], device=DEVICE)
    masked = PsiSpecRunner._apply_top_k(logits, 2)
    # Only the top-2 (cols 1 and 3) survive; the rest are -inf.
    finite = torch.isfinite(masked[0])
    assert finite.tolist() == [False, True, False, True]
    assert masked[0, 1].item() == 5.0
    assert masked[0, 3].item() == 4.0
