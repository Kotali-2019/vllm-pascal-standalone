# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Psi-Spec two-pass decode step (runner-independent orchestration).

This is the control flow of one Psi-Spec decode step: two causal forwards of
the *same* model over a small block, a KV pointer rollback between them, and
the :func:`vllm.v1.sample.psi_spec.psi_spec_verify` accept/reject. It is kept
free of any model-runner import so the block layout, the per-row LoRA gate, and
the KV-frontier arithmetic can be unit-tested against a fake forward.

The step is the *linear* path only: no tree attention, no FA3, no FP8. The
drafter is the base model with the Uno LoRA adapter gated **on** for the noise
rows and **off** for the seed row; the verifier is the base model (LoRA off).

KV frontier invariant (per request): ``num_cached_tokens == len(seq) - 1``. The
last committed token is the "uncached seed" at position ``C = num_cached``; it
is recomputed as the seed every step. A KV "rollback" is a pure pointer move --
discarded slots are simply overwritten by the next forward.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.v1.sample.psi_spec import psi_spec_verify

__all__ = ["PsiSpecTwoPassStep", "TwoPassInputs"]


@dataclass
class TwoPassInputs:
    """Per-step state and callables the step driver reads and writes.

    All tensors are on the model device. ``frontier`` is the per-request KV
    frontier ``C`` (number of cached tokens); ``seed`` is the token at position
    ``C`` (the uncached seed). ``forward`` runs one causal forward of the model
    over a ``[B, L]`` block and returns its logits as ``[B, L, V]``; it must
    apply ``lora_gate`` (a flat ``[B*L]`` 0/1 vector) as the per-row LoRA mask
    before the forward. ``sample`` draws one token per row of a ``[N, V]``
    logits tensor and must use the same (unfiltered, temperature) distribution
    as :func:`psi_spec_verify` so the accept/reject stays lossless.
    """

    frontier: torch.Tensor  # [B] int64
    seed: torch.Tensor  # [B] int64
    forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    sample: Callable[[torch.Tensor], torch.Tensor]
    set_frontier: Callable[[torch.Tensor], None]
    set_seed: Callable[[torch.Tensor], None]


class PsiSpecTwoPassStep:
    """Run one two-pass (draft -> verify -> accept/reject) decode step.

    Args:
        block_len: block length ``L`` (>= 1). Column 0 is the seed/clean row;
            columns 1..L-1 are the LoRA "spec" rows.
        mask_token_id: token id used to fill the noise rows when
            ``use_mask`` is set; otherwise noise is drawn from
            ``[noise_min, noise_max)``.
        temperature: sampling temperature; ``<= 0`` selects greedy verify.
        noise_min: inclusive lower bound of the random noise range.
        noise_max: exclusive upper bound of the random noise range.
        use_mask: if True, fill every noise row with ``mask_token_id`` instead
            of random noise.
        top_k: optional top-k support; forwarded to the verify so the
            accept/reject uses the same filtered distribution as the sampler.
        top_p: optional top-p nucleus; forwarded to the verify as above.
        device: device for the intermediate block tensors.
    """

    def __init__(
        self,
        block_len: int,
        mask_token_id: int,
        temperature: float = 1.0,
        noise_min: int = 1,
        noise_max: int | None = None,
        use_mask: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if block_len < 1:
            raise ValueError(f"block_len must be >= 1, got {block_len}")
        self.block_len = block_len
        self.mask_token_id = mask_token_id
        self.temperature = temperature
        self.noise_min = noise_min
        self.noise_max = (
            noise_max if noise_max is not None else max(noise_min + 1, mask_token_id)
        )
        self.use_mask = use_mask
        self.top_k = top_k
        self.top_p = top_p
        self.device = device

    def _make_noise(self, b: int, n: int) -> torch.Tensor:
        """Build the ``[B, n]`` noise block for the draft rows."""
        if n == 0:
            return torch.empty((b, 0), dtype=torch.long, device=self.device)
        if self.use_mask:
            return torch.full(
                (b, n), self.mask_token_id, dtype=torch.long, device=self.device
            )
        span = max(self.noise_max - self.noise_min, 1)
        return torch.randint(
            self.noise_min, self.noise_max, (b, n), device=self.device
        )

    def run(self, inp: TwoPassInputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one two-pass step.

        Args:
            inp: the per-step state and callables (see :class:`TwoPassInputs`).

        Returns:
            ``(committed, lengths)`` where ``committed`` is ``[B, L+1]`` (only
            the first ``lengths[b]`` entries are valid) and ``lengths`` is ``[B]``
            in ``[2, L+1]``. On return, ``inp.frontier``/``inp.seed`` (and the
            model's KV pointer) reflect the committed tokens for the next step.
        """
        b = inp.frontier.shape[0]
        l = self.block_len
        device = inp.frontier.device
        c = inp.frontier  # [B]
        arange_l = torch.arange(l, device=device)

        # ---- DRAFT: [seed, noise...] at [C, C+L), LoRA gate [0, 1, ...] ----
        draft_ids = torch.empty((b, l), dtype=torch.long, device=device)
        draft_ids[:, 0] = inp.seed
        if l > 1:
            draft_ids[:, 1:] = self._make_noise(b, l - 1)
        pos_draft = c.unsqueeze(1) + arange_l  # [B, L]
        gate_draft = torch.ones((b, l), dtype=torch.float32, device=device)
        gate_draft[:, 0] = 0.0
        draft_logits = inp.forward(draft_ids, pos_draft, gate_draft.reshape(-1))

        # clean = base-AR sample (seed row, LoRA off); specs = LoRA samples.
        clean = inp.sample(draft_logits[:, 0, :])  # [B]
        if l > 1:
            specs = inp.sample(
                draft_logits[:, 1:, :].reshape(-1, draft_logits.size(-1))
            )  # [B*(L-1)]
        else:
            specs = torch.empty((0,), dtype=torch.long, device=device)

        # Rollback: keep only the seed (position C); discard the noise rows.
        inp.set_frontier(c + 1)

        # ---- VERIFY: [clean, specs...] at [C+1, C+L+1), LoRA off ----
        verify_ids = torch.empty((b, l), dtype=torch.long, device=device)
        verify_ids[:, 0] = clean
        if l > 1:
            verify_ids[:, 1:] = specs.reshape(b, l - 1)
        pos_verify = (c + 1).unsqueeze(1) + arange_l  # [B, L]
        gate_verify = torch.zeros((b, l), dtype=torch.float32, device=device)
        verify_logits = inp.forward(verify_ids, pos_verify, gate_verify.reshape(-1))

        lookahead = inp.sample(verify_logits[:, -1, :])  # [B]

        # Assemble the proposal and the (verify, draft) logits in the layout
        # psi_spec_verify expects: col k (0..L-2) is the target/draft
        # distribution for spec k, col L-1 is the lookahead.
        proposal = torch.empty((b, l), dtype=torch.long, device=device)
        proposal[:, 0] = clean
        if l > 1:
            proposal[:, 1:] = specs.reshape(b, l - 1)
        if l > 1:
            draft_for_verify = torch.cat(
                [draft_logits[:, 1:, :], draft_logits[:, -1:, :]], dim=1
            )
        else:
            draft_for_verify = draft_logits

        committed, lengths = psi_spec_verify(
            proposal,
            verify_logits,
            draft_for_verify,
            lookahead,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

        # Rollback to the committed frontier and update the uncached seed.
        inp.set_frontier(c + lengths)
        new_seed = torch.gather(
            committed, 1, (lengths - 1).unsqueeze(1)
        ).squeeze(1)
        inp.set_seed(new_seed)
        return committed, lengths
