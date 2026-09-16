# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the Psi-Spec two-pass decode step orchestration.

The step is exercised against a *logit-oracle* fake forward: the model's
argmax at each block position is a value the test sets directly, so the whole
two-pass (draft block, verify block, accept/reject, KV rollback, seed update)
is hand-computable. This pins the block layout, the per-row LoRA gate, the
frontier arithmetic (``C -> C+1 -> C+len``), and the seed update.
"""

import pytest
import torch

from vllm.v1.sample.psi_spec_step import PsiSpecTwoPassStep, TwoPassInputs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V = 16


class _OracleForward:
    """A fake model whose argmax at block position j is a preset value.

    The first call is the draft forward (returns ``draft_oracle``), the second
    is the verify forward (returns ``verify_oracle``). It records the input
    ids, positions, and LoRA gate of each call for later assertions.
    """

    def __init__(self, draft_oracle: torch.Tensor, verify_oracle: torch.Tensor):
        self.draft_oracle = draft_oracle  # [B, L]
        self.verify_oracle = verify_oracle  # [B, L]
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def __call__(self, input_ids, positions, lora_gate):
        self.calls.append(
            (input_ids.clone(), positions.clone(), lora_gate.clone())
        )
        oracle = self.draft_oracle if len(self.calls) == 1 else self.verify_oracle
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], V),
            -10.0,
            dtype=torch.float32,
            device=DEVICE,
        )
        logits.scatter_(2, oracle.to(torch.long).unsqueeze(-1), 10.0)
        return logits


def _greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def _run_case(
    l: int,
    seed: int,
    c: int,
    draft_oracle: list[int],
    verify_oracle: list[int],
) -> dict:
    """Run one single-request step and return the observed end state."""
    step = PsiSpecTwoPassStep(block_len=l, mask_token_id=4, temperature=0.0, device=DEVICE)
    draft = torch.tensor([draft_oracle], dtype=torch.long, device=DEVICE)
    verify = torch.tensor([verify_oracle], dtype=torch.long, device=DEVICE)
    fwd = _OracleForward(draft, verify)

    state = {"frontier": torch.tensor([c], dtype=torch.long, device=DEVICE),
             "seed": torch.tensor([seed], dtype=torch.long, device=DEVICE)}

    def set_frontier(f):
        state["frontier"] = f

    def set_seed(s):
        state["seed"] = s

    inp = TwoPassInputs(
        frontier=state["frontier"],
        seed=state["seed"],
        forward=fwd,
        sample=_greedy_sample,
        set_frontier=set_frontier,
        set_seed=set_seed,
    )
    committed, lengths = step.run(inp)
    return {
        "committed": committed[0].tolist(),
        "lengths": int(lengths[0]),
        "frontier": int(state["frontier"][0]),
        "seed": int(state["seed"][0]),
        "calls": fwd.calls,
    }


def test_first_spec_rejected():
    """clean=11, spec0=6 rejected (target 12) -> [11, 12], len 2."""
    r = _run_case(3, seed=10, c=10, draft_oracle=[11, 6, 9], verify_oracle=[12, 7, 7])
    assert r["lengths"] == 2
    assert r["committed"][:2] == [11, 12]
    # Frontier: C -> C+1 (after draft) -> C+len (after verify) = 12.
    assert r["frontier"] == 12
    # New seed = last committed token.
    assert r["seed"] == 12


def test_all_accepted_includes_lookahead():
    """clean=11, spec0=12, spec1=13 all match targets -> + lookahead 14, len L+1."""
    r = _run_case(3, seed=10, c=10, draft_oracle=[11, 12, 13], verify_oracle=[12, 13, 14])
    assert r["lengths"] == 4  # L + 1
    assert r["committed"][:4] == [11, 12, 13, 14]
    assert r["frontier"] == 14  # C + len
    assert r["seed"] == 14


def test_accept_then_reject():
    """clean=11, spec0=12 accepted, spec1=5 rejected (target 13) -> [11,12,13], len 3."""
    r = _run_case(3, seed=10, c=10, draft_oracle=[11, 12, 5], verify_oracle=[12, 13, 14])
    assert r["lengths"] == 3
    assert r["committed"][:3] == [11, 12, 13]
    assert r["frontier"] == 13
    assert r["seed"] == 13


def test_block_layout_and_gates():
    """The draft block is [seed, ...] at [C, C+L) with gate [0,1,...]; the
    verify block is [clean, specs...] at [C+1, C+L+1) with gate all-0."""
    l = 4
    r = _run_case(l, seed=10, c=10, draft_oracle=[11, 12, 13, 14], verify_oracle=[12, 13, 14, 15])
    assert len(r["calls"]) == 2
    draft_ids, draft_pos, draft_gate = r["calls"][0]
    verify_ids, verify_pos, verify_gate = r["calls"][1]

    # Draft: seed first, positions [C, C+L), gate off on the seed row only.
    assert draft_ids[0, 0].item() == 10
    assert draft_pos[0].tolist() == [10, 11, 12, 13]
    assert draft_gate.tolist() == [0.0, 1.0, 1.0, 1.0]

    # Verify: clean (draft col 0) first, then the specs; positions shifted by 1;
    # LoRA fully off.
    assert verify_ids[0, 0].item() == 11  # clean
    assert verify_ids[0, 1:].tolist() == [12, 13, 14]  # specs
    assert verify_pos[0].tolist() == [11, 12, 13, 14]
    assert verify_gate.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_ar_no_specs():
    """L == 1: commit clean + lookahead, length 2 (plain AR step)."""
    r = _run_case(1, seed=10, c=10, draft_oracle=[11], verify_oracle=[12])
    assert r["lengths"] == 2
    assert r["committed"][:2] == [11, 12]
    assert r["frontier"] == 12
    assert r["seed"] == 12


def test_batch_independent_rows():
    """Two requests advance independently; each frontier/seed tracks its own len."""
    l = 3
    step = PsiSpecTwoPassStep(block_len=l, mask_token_id=4, temperature=0.0, device=DEVICE)
    # req 0: all accepted (len 4); req 1: first spec rejected (len 2).
    draft = torch.tensor([[11, 12, 13], [11, 6, 9]], dtype=torch.long, device=DEVICE)
    verify = torch.tensor([[12, 13, 14], [12, 7, 7]], dtype=torch.long, device=DEVICE)
    fwd = _OracleForward(draft, verify)
    state = {
        "frontier": torch.tensor([10, 20], dtype=torch.long, device=DEVICE),
        "seed": torch.tensor([10, 30], dtype=torch.long, device=DEVICE),
    }
    inp = TwoPassInputs(
        frontier=state["frontier"],
        seed=state["seed"],
        forward=fwd,
        sample=_greedy_sample,
        set_frontier=lambda f: state.__setitem__("frontier", f),
        set_seed=lambda s: state.__setitem__("seed", s),
    )
    committed, lengths = step.run(inp)
    assert lengths.tolist() == [4, 2]
    assert committed[0, :4].tolist() == [11, 12, 13, 14]
    assert committed[1, :2].tolist() == [11, 12]
    assert state["frontier"].tolist() == [14, 22]  # C + len per request
    assert state["seed"].tolist() == [14, 12]


@pytest.mark.parametrize("l", [1, 2, 3, 8])
def test_length_bounds(l: int):
    """Committed length is always in [2, L+1] regardless of the oracle values."""
    torch.manual_seed(0)
    step = PsiSpecTwoPassStep(block_len=l, mask_token_id=4, temperature=0.0, device=DEVICE)
    b = 3
    draft = torch.randint(0, V, (b, l), device=DEVICE)
    verify = torch.randint(0, V, (b, l), device=DEVICE)
    fwd = _OracleForward(draft, verify)
    state = {
        "frontier": torch.arange(b, dtype=torch.long, device=DEVICE) * 7,
        "seed": torch.randint(0, V, (b,), device=DEVICE),
    }
    inp = TwoPassInputs(
        frontier=state["frontier"],
        seed=state["seed"],
        forward=fwd,
        sample=_greedy_sample,
        set_frontier=lambda f: state.__setitem__("frontier", f),
        set_seed=lambda s: state.__setitem__("seed", s),
    )
    _, lengths = step.run(inp)
    assert (lengths >= 2).all()
    assert (lengths <= l + 1).all()
