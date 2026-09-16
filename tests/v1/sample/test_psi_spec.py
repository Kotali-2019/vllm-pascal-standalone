# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the Psi-Spec two-pass accept/reject (``psi_spec_verify``).

Each test compares the module output against an independent, per-row Python
reference that implements the standard speculative-sampling accept/reject. The
reference uses the same ``accept_random`` / ``correction_random`` draws so the
comparison is exact (no statistical tolerance needed).
"""

import bisect

import pytest
import torch

from vllm.v1.sample.psi_spec import psi_spec_verify

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Independent reference (pure Python over full distributions)
# ---------------------------------------------------------------------------


def _ref_support(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    vocab: int,
) -> tuple[list[int], list[float]]:
    """Build the target/draft support ``(ids, probs)`` in the module's order.

    Matches ``psi_spec``'s distribution builders: top-k (renormalized over the k
    support), top-p (nucleus prefix of the full softmax, not renormalized), or
    the full vocab in token-id order for the unfiltered case.
    """
    scaled = logits / temperature
    if top_k is not None and 0 < top_k < vocab:
        vals, idx = torch.topk(scaled, top_k, dim=-1)
        return idx.tolist(), torch.softmax(vals, dim=-1).tolist()
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_vals, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_vals, dim=-1)
        cum = sorted_probs.cumsum(dim=-1)
        over = cum > top_p
        first_over = int(over.to(torch.int64).argmax())
        keep = first_over + 1 if bool(over.any()) else vocab
        return sorted_idx[:keep].tolist(), sorted_probs[:keep].tolist()
    return list(range(vocab)), torch.softmax(scaled, dim=-1).tolist()


def _prob_of(ids: list[int], probs: list[float], token: int) -> float:
    for i, t in enumerate(ids):
        if t == token:
            return probs[i]
    return 0.0


def _sample_inverse_cdf(probs: list[float], r: float) -> int:
    cdf = []
    acc = 0.0
    for x in probs:
        acc += x
        cdf.append(acc)
    return min(bisect.bisect_right(cdf, r), len(probs) - 1)


def _ref_committed(
    proposal: torch.Tensor,
    verify_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    lookahead: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    accept_random: torch.Tensor,
    correction_random: torch.Tensor,
) -> list[list[int]]:
    """Independent per-request reference. Returns committed token lists.

    The accept/reject math and committed-sequence construction are computed from
    scratch; the correction is sampled in the target-support space (matching the
    module and the nano reference) then mapped back to a token id.
    """
    b, l = proposal.shape
    vocab = verify_logits.size(-1)
    n_spec = l - 1
    verify_flat = verify_logits[:, :n_spec, :].reshape(-1, vocab)
    draft_flat = draft_logits[:, :n_spec, :].reshape(-1, vocab)
    spec_vals = proposal[:, 1:l].reshape(-1)

    committed_all = []
    for req in range(b):
        clean = int(proposal[req, 0])
        row = [clean]
        rejected = False
        for j in range(n_spec):
            n = req * n_spec + j
            spec = int(spec_vals[n])
            v, d = verify_flat[n], draft_flat[n]
            if temperature <= 0:
                target = int(v.argmax())
                if spec == target:
                    row.append(spec)
                    continue
                row.append(target)
                rejected = True
                break
            p_ids, p_probs = _ref_support(v, temperature, top_k, top_p, vocab)
            q_ids, q_probs = _ref_support(d, temperature, top_k, top_p, vocab)
            p_spec = _prob_of(p_ids, p_probs, spec)
            q_spec = _prob_of(q_ids, q_probs, spec)
            ratio = min(1.0, p_spec / q_spec) if q_spec > 0 else 0.0
            if accept_random[n].item() < ratio:
                row.append(spec)
                continue
            corr = [max(0.0, p_probs[k] - _prob_of(q_ids, q_probs, p_ids[k])) for k in range(len(p_ids))]
            s = sum(corr)
            corr = [x / s for x in corr] if s > 0 else list(p_probs)
            row.append(p_ids[_sample_inverse_cdf(corr, correction_random[n].item())])
            rejected = True
            break
        if not rejected:
            row.append(int(lookahead[req]))
        committed_all.append(row)
    return committed_all


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_inputs(b: int, l: int, vocab: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    proposal = torch.randint(0, vocab, (b, l), generator=g)
    verify_logits = torch.randn(b, l, vocab, generator=g)
    draft_logits = torch.randn(b, l, vocab, generator=g)
    lookahead = torch.randint(0, vocab, (b,), generator=g)
    return proposal, verify_logits, draft_logits, lookahead


def _run_and_check(
    b: int,
    l: int,
    vocab: int,
    seed: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
):
    proposal, verify_logits, draft_logits, lookahead = _make_inputs(b, l, vocab, seed)
    n = b * (l - 1)
    g = torch.Generator(device="cpu").manual_seed(seed + 1000)
    accept_random = torch.rand(n, generator=g)
    correction_random = torch.rand(n, generator=g)

    committed, lengths = psi_spec_verify(
        proposal,
        verify_logits,
        draft_logits,
        lookahead,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        accept_random=accept_random,
        correction_random=correction_random,
    )
    ref = _ref_committed(
        proposal,
        verify_logits,
        draft_logits,
        lookahead,
        temperature,
        top_k,
        top_p,
        accept_random,
        correction_random,
    )
    for req in range(b):
        got = [int(x) for x in committed[req, : lengths[req]].tolist()]
        assert got == ref[req], f"req {req}: got {got}, ref {ref[req]} (len {int(lengths[req])})"
        # Length invariant: 2 (first spec rejected) .. L+1 (all accepted + lookahead).
        assert 2 <= int(lengths[req]) <= l + 1


@pytest.mark.parametrize("l", [2, 4, 8])
def test_greedy(l: int):
    _run_and_check(b=3, l=l, vocab=32, seed=1, temperature=0.0, top_k=None, top_p=None)


@pytest.mark.parametrize("l", [2, 4, 8])
def test_unfiltered_stochastic(l: int):
    _run_and_check(b=3, l=l, vocab=32, seed=2, temperature=0.8, top_k=None, top_p=None)


@pytest.mark.parametrize("l", [2, 4])
def test_top_k(l: int):
    _run_and_check(b=3, l=l, vocab=32, seed=3, temperature=0.9, top_k=8, top_p=None)


@pytest.mark.parametrize("l", [2, 4])
def test_top_p(l: int):
    _run_and_check(b=3, l=l, vocab=32, seed=4, temperature=0.9, top_k=None, top_p=0.9)


def test_ar_no_specs():
    """L == 1: commit clean + lookahead, length 2."""
    proposal = torch.tensor([[5], [7]])
    verify_logits = torch.randn(2, 1, 16)
    draft_logits = torch.randn(2, 1, 16)
    lookahead = torch.tensor([3, 9])
    committed, lengths = psi_spec_verify(proposal, verify_logits, draft_logits, lookahead)
    assert (lengths == 2).all()
    assert committed[0].tolist() == [5, 3]
    assert committed[1].tolist() == [7, 9]


def test_all_accepted_includes_lookahead():
    """When every spec matches the target argmax (greedy), length == L+1."""
    vocab, l = 16, 5
    proposal = torch.arange(l, dtype=torch.long).unsqueeze(0)  # [0,1,2,3,4]
    # Target argmax at each spec position j is proposal[j+1].
    verify_logits = torch.full((1, l, vocab), -10.0)
    for j in range(l - 1):
        verify_logits[0, j, proposal[0, j + 1]] = 10.0
    verify_logits[0, l - 1, 12] = 10.0  # lookahead position
    draft_logits = verify_logits.clone()
    lookahead = torch.tensor([12])
    committed, lengths = psi_spec_verify(
        proposal, verify_logits, draft_logits, lookahead, temperature=0.0
    )
    assert int(lengths[0]) == l + 1
    assert committed[0].tolist() == [0, 1, 2, 3, 4, 12]


def test_first_spec_rejected():
    """First spec mismatched under greedy -> length 2, correction = target argmax."""
    vocab, l = 16, 4
    proposal = torch.tensor([[0, 5, 6, 7]])
    verify_logits = torch.full((1, l, vocab), -10.0)
    verify_logits[0, 0, 5] = 10.0  # target says 5 (matches spec0)
    verify_logits[0, 1, 9] = 10.0  # target says 9 (spec1 is 6 -> reject)
    verify_logits[0, 2, 6] = 10.0
    verify_logits[0, 3, 12] = 10.0
    draft_logits = verify_logits.clone()
    lookahead = torch.tensor([12])
    committed, lengths = psi_spec_verify(
        proposal, verify_logits, draft_logits, lookahead, temperature=0.0
    )
    # clean=0, spec0=5 accepted, spec1=6 rejected -> correction 9, stop.
    assert int(lengths[0]) == 3
    assert committed[0, :3].tolist() == [0, 5, 9]
