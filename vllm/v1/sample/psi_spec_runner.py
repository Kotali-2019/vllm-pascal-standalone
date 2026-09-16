# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Psi-Spec two-pass decode driver (vLLM-native, GPU model runner).

Binds the runner-independent two-pass step (:class:`PsiSpecTwoPassStep`) to a
live :class:`~vllm.v1.worker.gpu_model_runner.GPUModelRunner`. Each two-pass
forward is a "prefill of ``L`` tokens with a prefix": it drives the runner's
real forward machinery (``_prepare_inputs`` + ``_build_attention_metadata`` +
``_get_slot_mappings`` + ``_model_forward`` + ``compute_logits``) over a small
block at an arbitrary KV frontier, applies the per-row LoRA gate (C1), and
rolls the KV frontier back with a pure pointer move between the draft and
verify forwards.

The driver is the *linear* path only (no tree attention, no FA3, no FP8). The
drafter is the base model with the Uno LoRA adapter gated **on** for the noise
rows and **off** for the seed row; the verifier is the base model (LoRA off).
Block allocation is left to the engine's scheduler, exactly as in the normal
decode flow: the driver assumes the block table already covers the span it
writes, so a KV "rollback" is a pointer move and the discarded slots are
simply overwritten by the next forward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.forward_context import set_forward_context
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.psi_spec_step import PsiSpecTwoPassStep, TwoPassInputs

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

__all__ = ["PsiSpecRunner"]


class PsiSpecRunner:
    """Drive the Psi-Spec two-pass decode against a live GPU model runner.

    The runner must already hold the requests to step (added by the engine's
    prefill) in ``runner.input_batch``; the driver operates on rows
    ``0 .. num_reqs-1`` in that order. It owns the per-row two-pass state
    (the KV frontier ``C`` and the uncached seed token) and exposes one
    :meth:`step` that runs a full draft -> verify -> accept/reject round.

    Args:
        runner: the :class:`GPUModelRunner` to drive.
        block_len: block length ``L`` (>= 1). Column 0 is the seed/clean row;
            columns 1..L-1 are the LoRA "spec" rows.
        mask_token_id: token id used to fill the noise rows when ``use_mask``
            is set; otherwise noise is drawn from ``[noise_min, noise_max)``.
        temperature: sampling temperature; ``<= 0`` selects greedy verify.
        noise_min: inclusive lower bound of the random noise range.
        noise_max: exclusive upper bound of the random noise range.
        use_mask: if True, fill every noise row with ``mask_token_id`` instead
            of random noise.
        top_k: optional top-k support for the sampler and the verify.
        top_p: optional top-p nucleus for the sampler and the verify.
    """

    def __init__(
        self,
        runner: "GPUModelRunner",
        block_len: int,
        mask_token_id: int,
        temperature: float = 1.0,
        noise_min: int = 1,
        noise_max: int | None = None,
        use_mask: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> None:
        self.runner = runner
        self.device = runner.device
        self.block_len = block_len
        self.vocab_size = runner.model_config.get_vocab_size()
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self._step = PsiSpecTwoPassStep(
            block_len=block_len,
            mask_token_id=mask_token_id,
            temperature=temperature,
            noise_min=noise_min,
            noise_max=noise_max,
            use_mask=use_mask,
            top_k=top_k,
            top_p=top_p,
            device=self.device,
        )
        # Per-row two-pass state, keyed by input_batch row index.
        self._frontier: dict[int, int] = {}
        self._seed: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Per-row state
    # ------------------------------------------------------------------

    def set_request_state(self, row: int, frontier: int, seed: int) -> None:
        """Record the KV frontier ``C`` and uncached seed for one row."""
        self._frontier[row] = frontier
        self._seed[row] = seed

    # ------------------------------------------------------------------
    # The two-pass forward (draft or verify block)
    # ------------------------------------------------------------------

    def _forward_block(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        lora_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Run one causal forward of the model over a ``[B, L]`` block.

        ``input_ids``/``positions`` are the block tokens and their absolute
        positions (``positions[:, 0]`` is the block's KV frontier); ``lora_gate``
        is a flat ``[B*L]`` 0/1 vector applied as the per-row LoRA mask (C1).
        Returns the block logits as ``[B, L, V]`` (one row per block position).
        """
        runner = self.runner
        ib = runner.input_batch
        b, l = input_ids.shape
        n = b * l

        # 1. Point each row's KV frontier at the block start and write the
        #    block tokens into the input batch at that frontier. ``token_ids_cpu``
        #    is a CPU numpy view, so the (device) block is moved to host first.
        for r in range(b):
            c = int(positions[r, 0])
            ib.num_computed_tokens_cpu[r] = c
            ib.token_ids_cpu[r, c : c + l] = input_ids[r].cpu().numpy()

        # 2. Minimal SchedulerOutput: every row schedules ``l`` tokens.
        so = SchedulerOutput.make_empty()
        so.num_scheduled_tokens = {
            ib.req_ids[r]: l for r in range(b)
        }
        so.total_num_scheduled_tokens = n
        num_scheduled_tokens_np = np.full(b, l, dtype=np.int32)
        num_sampled_tokens_np = np.full(b, l, dtype=np.int32)

        # 3. Activate the request's LoRA adapter for this forward (no-op when
        #    LoRA is off), then apply the per-row gate (C1) that zeroes the
        #    seed row so only the noise rows see the adapter.
        wrapper = self._punica_wrapper()
        if wrapper is not None and hasattr(runner, "set_active_loras"):
            runner.set_active_loras(
                ib, num_scheduled_tokens_np, num_sampled_tokens_np
            )
        if wrapper is not None:
            wrapper.lora_row_mask = lora_gate.to(device=self.device)

        # 4. Positions / input_ids / slot mapping / seq_lens.
        runner._prepare_inputs(so, num_scheduled_tokens_np)

        # 5. Slot mappings (by KV group and by layer) + attention metadata.
        slot_mappings_by_group, slot_mappings = runner._get_slot_mappings(
            num_tokens_padded=n, num_reqs_padded=b, num_tokens_unpadded=n
        )
        attn_metadata, _ = runner._build_attention_metadata(
            num_tokens=n,
            num_reqs=b,
            max_query_len=l,
            num_scheduled_tokens=so.num_scheduled_tokens,
            slot_mappings=slot_mappings_by_group,
        )

        # 6. Forward over the whole block (all positions, not just the last).
        input_ids_gpu = runner.input_ids.gpu[:n]
        positions_gpu = runner.positions[:n]
        with set_forward_context(
            attn_metadata,
            runner.vllm_config,
            num_tokens=n,
            slot_mapping=slot_mappings,
        ):
            hidden_states = runner._model_forward(
                input_ids=input_ids_gpu, positions=positions_gpu
            )

        # 7. Logits at every block position -> [B, L, V].
        logits = runner.model.compute_logits(hidden_states).reshape(b, l, -1)

        # 8. Clear the gate so it does not leak into the next forward.
        if wrapper is not None:
            wrapper.lora_row_mask = None
        return logits

    def _punica_wrapper(self):
        """The language-model LoRA wrapper, or None if LoRA is not active."""
        if not hasattr(self.runner, "get_lora_punica_wrapper"):
            return None
        return self.runner.get_lora_punica_wrapper()

    # ------------------------------------------------------------------
    # Sampling (must match the verify distribution for the lossless property)
    # ------------------------------------------------------------------

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Draw one token per row of a ``[N, V]`` logits tensor.

        Uses the same (temperature, top-k, top-p) as
        :func:`vllm.v1.sample.psi_spec.psi_spec_verify` so the accept/reject
        stays lossless.
        """
        if self.temperature <= 0:
            return logits.argmax(dim=-1)
        scaled = logits / self.temperature
        if self.top_k is not None and 0 < self.top_k < self.vocab_size:
            scaled = self._apply_top_k(scaled, self.top_k)
        elif self.top_p is not None and 0.0 < self.top_p < 1.0:
            scaled = self._apply_top_p(scaled, self.top_p)
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Zero out all but the top-``k`` logits per row."""
        idx = torch.topk(logits, k, dim=-1).indices
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(1, idx, 0.0)
        return logits + mask

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Zero out the nucleus complement per row (nucleus sampling)."""
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum = sorted_probs.cumsum(dim=-1)
        # Remove the first token that pushes the cumulative past ``p``.
        over = (cum - sorted_probs) > p
        remove = over.roll(1, dims=-1)
        remove[..., 0] = False
        sorted_mask = torch.where(remove, float("-inf"), 0.0)
        return logits + sorted_mask.scatter(1, sorted_idx, sorted_mask)

    # ------------------------------------------------------------------
    # One two-pass decode step
    # ------------------------------------------------------------------

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one draft -> verify -> accept/reject round over the batch.

        Returns:
            ``(committed, lengths)`` where ``committed`` is ``[B, L+1]`` (only
            the first ``lengths[b]`` entries are valid) and ``lengths`` is ``[B]``
            in ``[2, L+1]``. On return the per-row frontier/seed reflect the
            committed tokens for the next step.
        """
        runner = self.runner
        b = runner.input_batch.num_reqs
        if b == 0:
            raise ValueError("no requests in the input batch to step")
        device = self.device

        frontier = torch.tensor(
            [self._frontier[r] for r in range(b)], dtype=torch.long, device=device
        )
        seed = torch.tensor(
            [self._seed[r] for r in range(b)], dtype=torch.long, device=device
        )

        def set_frontier(f: torch.Tensor) -> None:
            for r in range(b):
                self._frontier[r] = int(f[r])

        def set_seed(s: torch.Tensor) -> None:
            for r in range(b):
                self._seed[r] = int(s[r])

        inp = TwoPassInputs(
            frontier=frontier,
            seed=seed,
            forward=self._forward_block,
            sample=self._sample,
            set_frontier=set_frontier,
            set_seed=set_seed,
        )
        committed, lengths = self._step.run(inp)
        return committed, lengths
