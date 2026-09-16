# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Psi-Spec two-pass accept/reject (pure torch, runner-independent).

This is the sampling core of the Psi-Spec two-pass decode step. Given the
proposal block (a base-AR "clean" token plus LoRA "spec" tokens), the target
``verify_logits`` (base model), the draft ``draft_logits`` (LoRA model), and a
sampled lookahead token, it applies standard speculative-sampling accept/reject
and returns the committed token rows plus their lengths.

The module is intentionally free of any model-runner dependency so it can be
unit-tested in isolation. It is the *linear* path only: no tree attention, no
FA3, no FP8. The unfiltered (full-softmax) and filtered (top-k/top-p) target
and draft distributions are built here from raw logits.
"""

from __future__ import annotations

import torch

__all__ = ["psi_spec_verify"]


def _top_k_probs(
    logits: torch.Tensor,
    k: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the top-``k`` support ``(ids, probs)`` of ``softmax(logits/T)``.

    ``ids`` and ``probs`` are both ``[N, k]``; ``probs`` is normalized over the
    kept support only.
    """
    scaled = logits / temperature
    vals, ids = torch.topk(scaled, k, dim=-1)
    probs = torch.softmax(vals, dim=-1)
    return ids, probs


def _top_p_probs(
    logits: torch.Tensor,
    p: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the top-``p`` nucleus as a fixed-width ``(ids, probs)`` pair.

    Each row keeps the smallest prefix whose cumulative probability reaches
    ``p`` (at least one token). Rows are padded to the batch's maximum nucleus
    width with ``id = -1`` and ``prob = 0`` so the result is a rectangular
    ``[N, K]`` tensor. This is the reference implementation; the production
    two-pass uses a fused kernel instead.
    """
    scaled = logits / temperature
    sorted_vals, sorted_ids = torch.sort(scaled, dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_vals, dim=-1)
    cum = sorted_probs.cumsum(dim=-1)
    over = cum > p
    has_over = over.any(dim=-1)
    first_over = over.to(torch.int64).argmax(dim=-1)
    keep = torch.where(has_over, first_over + 1, torch.full_like(first_over, logits.size(-1)))
    # Host sync to size the padded tensor to the widest nucleus in the batch.
    k = max(int(keep.max().item()), 1)
    pos = torch.arange(k, device=logits.device).unsqueeze(0)  # [1, K]
    pos_mask = pos < keep.unsqueeze(1)  # [N, K]
    gather_idx = pos.expand(logits.size(0), k).clamp(max=logits.size(-1) - 1)
    ids = torch.where(pos_mask, sorted_ids.gather(1, gather_idx), torch.full_like(sorted_ids[:, :k], -1))
    probs = torch.where(pos_mask, sorted_probs.gather(1, gather_idx), torch.zeros_like(sorted_probs[:, :k]))
    return ids, probs


def _verify_rows(
    p_ids: torch.Tensor | None,
    p_probs: torch.Tensor,
    q_ids: torch.Tensor | None,
    q_probs: torch.Tensor,
    spec_vals: torch.Tensor,
    accept_random: torch.Tensor,
    full: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept/reject each proposed token against the target distribution.

    ``p`` is the target (verify) distribution and ``q`` the draft distribution,
    each given as either a full ``[N, V]`` probability row (``full=True``) or a
    sparse ``(ids, probs)`` support pair. Returns the per-row accept flag and
    the correction distribution (over the target support) used on rejection.
    """
    if full:
        p_spec = p_probs.gather(1, spec_vals.unsqueeze(1)).squeeze(1)
        q_spec = q_probs.gather(1, spec_vals.unsqueeze(1)).squeeze(1)
        q_on_clean = q_probs
    else:
        p_spec = torch.where(p_ids.eq(spec_vals.unsqueeze(1)), p_probs, torch.zeros_like(p_probs)).sum(1)
        q_spec = torch.where(q_ids.eq(spec_vals.unsqueeze(1)), q_probs, torch.zeros_like(q_probs)).sum(1)
        q_on_clean = torch.where(
            p_ids.unsqueeze(2).eq(q_ids.unsqueeze(1)),
            q_probs.unsqueeze(1),
            torch.zeros((), dtype=q_probs.dtype, device=q_probs.device),
        ).sum(2)
    ratio = torch.where(q_spec > 0, p_spec / q_spec, torch.zeros_like(p_spec))
    accepted = accept_random.lt(torch.clamp(ratio, max=1.0))
    correction = torch.clamp(p_probs - q_on_clean, min=0.0)
    csum = correction.sum(1, keepdim=True)
    correction = torch.where(csum > 0, correction / csum.clamp_min(1e-12), p_probs)
    return accepted, correction


def _sample_from_probs(probs: torch.Tensor, random: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF sample one index per row of ``probs`` using ``random``."""
    cdf = probs.cumsum(dim=-1)
    idx = torch.searchsorted(cdf, random.unsqueeze(-1), right=True).squeeze(-1)
    return idx.clamp(max=probs.size(-1) - 1)


def _build_committed_payload(
    proposal: torch.Tensor,
    accepted: torch.Tensor,
    correction_tokens: torch.Tensor,
    lookahead: torch.Tensor,
) -> torch.Tensor:
    """Assemble ``[committed..., length]`` rows from the accept/reject result.

    ``proposal`` is ``[B, L]`` (col 0 = clean, cols 1..L-1 = specs), ``accepted``
    and ``correction_tokens`` are ``[B, L-1]``, ``lookahead`` is ``[B]``. The
    returned tensor is ``[B, L+2]``: cols ``0..L`` hold the committed tokens
    (clean, accepted specs, one correction at the first reject, lookahead) and
    the last col holds the committed length (``2..L+1``).
    """
    b, l = proposal.shape
    device = proposal.device
    n_spec = l - 1
    max_commit = l + 1
    accepted = accepted.to(device=device, dtype=torch.bool)
    correction_tokens = correction_tokens.to(device=device, dtype=proposal.dtype)
    first_reject = torch.zeros((b,), dtype=torch.long, device=device)
    reject_mask = ~accepted
    rejected = reject_mask.any(dim=1)
    payload = torch.empty((b, max_commit + 1), dtype=proposal.dtype, device=device)
    committed = payload[:, :max_commit]
    committed[:, :l] = proposal
    if n_spec > 0:
        first_reject = torch.argmax(reject_mask.to(torch.int64), dim=1)
        reject_cols = first_reject + 1
        replacement = correction_tokens.gather(1, first_reject.unsqueeze(1)).squeeze(1)
        original = committed.gather(1, reject_cols.unsqueeze(1)).squeeze(1)
        committed.scatter_(
            1,
            reject_cols.unsqueeze(1),
            torch.where(rejected, replacement, original).unsqueeze(1),
        )
    full_length = torch.full((b,), l, dtype=torch.long, device=device)
    lengths = torch.where(rejected, first_reject + 2, full_length)
    committed[:, l] = lookahead.to(device=device, dtype=proposal.dtype).reshape(b)
    lengths = lengths + (~rejected).to(lengths.dtype)
    payload[:, -1] = lengths
    return payload


def psi_spec_verify(
    proposal: torch.Tensor,
    verify_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    lookahead: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    accept_random: torch.Tensor | None = None,
    correction_random: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the Psi-Spec accept/reject for one decode step.

    Args:
        proposal: ``[B, L]`` int tokens. Col 0 is the base-AR "clean" token;
            cols 1..L-1 are the LoRA "spec" proposals.
        verify_logits: ``[B, L, V]`` target (base model) logits.
        draft_logits: ``[B, L, V]`` draft (LoRA model) logits.
        lookahead: ``[B]`` int token sampled from the last verify position.
        temperature: sampling temperature; ``<= 0`` selects greedy verify.
        top_k: optional top-k support for the filtered path.
        top_p: optional top-p nucleus for the filtered path.
        accept_random: optional ``[B*(L-1)]`` uniform draw for determinism.
        correction_random: optional ``[B*(L-1)]`` uniform draw for determinism.

    Returns:
        ``(committed, lengths)`` where ``committed`` is ``[B, L+1]`` (only the
        first ``lengths[b]`` entries are valid) and ``lengths`` is ``[B]``.
    """
    b, l = proposal.shape
    vocab = verify_logits.size(-1)
    device = verify_logits.device
    n_spec = l - 1

    if n_spec <= 0:
        # No specs: commit the clean token plus the lookahead (plain AR step).
        committed = torch.empty((b, 2), dtype=proposal.dtype, device=device)
        committed[:, 0] = proposal[:, 0]
        committed[:, 1] = lookahead.to(device=device, dtype=proposal.dtype)
        lengths = torch.full((b,), 2, dtype=torch.long, device=device)
        return committed, lengths

    verify_flat = verify_logits[:, :n_spec, :].reshape(-1, vocab)
    draft_flat = draft_logits[:, :n_spec, :].reshape(-1, vocab)
    spec_vals = proposal[:, 1:l].reshape(-1).to(device=device)
    n = verify_flat.size(0)
    if accept_random is None:
        accept_random = torch.rand(n, dtype=verify_flat.dtype, device=device)

    if temperature <= 0:
        # Greedy verify: accept a spec only if it matches the target argmax.
        target = verify_flat.argmax(dim=-1)
        accepted = spec_vals == target
        correction_tokens = target
    else:
        use_top_k = top_k is not None and 0 < top_k < vocab
        use_top_p = top_p is not None and 0.0 < top_p < 1.0
        if use_top_k or use_top_p:
            if use_top_k:
                p_ids, p_probs = _top_k_probs(verify_flat, top_k, temperature)
                q_ids, q_probs = _top_k_probs(draft_flat, top_k, temperature)
            else:
                p_ids, p_probs = _top_p_probs(verify_flat, top_p, temperature)
                q_ids, q_probs = _top_p_probs(draft_flat, top_p, temperature)
            accepted, correction_probs = _verify_rows(
                p_ids, p_probs, q_ids, q_probs, spec_vals, accept_random, full=False
            )
            if correction_random is None:
                correction_random = torch.rand(n, dtype=correction_probs.dtype, device=device)
            corr_idx = _sample_from_probs(correction_probs, correction_random)
            correction_tokens = p_ids.gather(1, corr_idx.unsqueeze(1)).squeeze(1)
        else:
            p_probs = torch.softmax(verify_flat / temperature, dim=-1)
            q_probs = torch.softmax(draft_flat / temperature, dim=-1)
            accepted, correction_probs = _verify_rows(
                None, p_probs, None, q_probs, spec_vals, accept_random, full=True
            )
            if correction_random is None:
                correction_random = torch.rand(n, dtype=correction_probs.dtype, device=device)
            correction_tokens = _sample_from_probs(correction_probs, correction_random)

    accepted = accepted.reshape(b, n_spec)
    correction_tokens = correction_tokens.reshape(b, n_spec)
    payload = _build_committed_payload(proposal, accepted, correction_tokens, lookahead)
    return payload[:, : l + 1], payload[:, -1]
