# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Psi-Spec two-pass decode CLI (vLLM-native, Uno gated LoRA).

A port of ``/root/uno/inference.py``: it loads a base model plus the Uno gated
LoRA adapter and runs the lossless two-pass decode (draft with the adapter,
verify with the base) through vLLM's real model + KV + attention, then prints
the generations and, with ``--compare-ar``, a decode-throughput comparison
against plain autoregressive decoding.

The engine is built in-process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) with the
V1 model runner and the Triton attention backend so the driver can reach the
live ``GPUModelRunner`` directly. In a dedicated engine the whole KV pool is
free, so each prompt's request is seeded by hand and stepped with
:class:`~vllm.v1.sample.psi_spec_runner.PsiSpecRunner`.

The two-pass is the *linear* path only (no tree attention, no FA3, no FP8) and
targets a single GPU (``--tensor-parallel-size 1``); the 0.9B model fits on one
card. In the greedy limit the output is a sequence of base-AR argmaxes and so
matches a plain greedy run token-for-token; under sampling it is distributed
exactly like the base model (the verify is the base model).

Run::

    python -m vllm.v1.sample.psi_spec_cli \
        --model IFM/K2-Horizon-0.9B \
        --gated-lora-path /path/to/K2-Horizon-0.9B-Uno \
        --prompt "Say the word hello." \
        --compare-ar
"""

from __future__ import annotations

import os

# In-process engine, V1 runner, Triton attention: the config the driver needs to
# reach the live model runner directly (Pascal-safe).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

import argparse
import json
import time

import torch

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.v1.sample.psi_spec_runner import PsiSpecRunner
from vllm.v1.worker.gpu_input_batch import CachedRequestState

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (mirrors ``/root/uno/inference.py``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="base model (HF id or path)")
    parser.add_argument("--gated-lora-path", required=True, help="Uno gated LoRA dir")
    parser.add_argument(
        "--mode",
        choices=("twopass", "direct"),
        default="twopass",
        help=(
            "twopass: lossless two-pass (LoRA draft, base verify); "
            "direct: apply the Uno LoRA to the whole sequence (no two-pass)"
        ),
    )
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--diffusion-block-size", type=int, default=8)
    parser.add_argument(
        "--noise-mode",
        choices=("random_uniform", "mask"),
        default="random_uniform",
        help="draft-row fill: random vocab tokens or the mask token",
    )
    parser.add_argument("--mask-token-id", type=int, default=0)
    parser.add_argument("--stop-token-ids", type=int, nargs="*", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--compare-ar",
        action="store_true",
        help="also run a plain-AR baseline to report the decode speedup",
    )
    return parser


def _seed_request(
    runner, lora_req: LoRARequest, prompt_ids: list[int], max_tokens: int, temperature: float
) -> CachedRequestState:
    """Seed one request into the runner's input batch (whole KV pool is free).

    Replicates the model runner's registration (request-state dict + input
    batch + condense + refresh) so the driver's manual forward sees it.
    """
    block_size = runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
    max_model_len = runner.model_config.max_model_len
    n_blocks = (max_model_len + block_size - 1) // block_size
    req = CachedRequestState(
        req_id="psi0",
        prompt_token_ids=prompt_ids,
        mm_features=[],
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=temperature),
        generator=None,
        block_ids=(list(range(n_blocks)),),
        num_computed_tokens=0,
        output_token_ids=[],
        lora_request=lora_req,
    )
    runner.requests[req.req_id] = req
    runner.input_batch.add_request(req)
    runner.input_batch.condense()
    runner.input_batch.refresh_metadata()
    return req


def _clear_request(runner, req_id: str) -> None:
    """Tear down a manually-seeded request (inverse of :func:`_seed_request`)."""
    runner.input_batch.remove_request(req_id)
    runner.input_batch.condense()
    runner.requests.pop(req_id, None)


def _two_pass_tokens(
    runner, lora_req: LoRARequest, prompt_ids: list[int], args, max_steps: int | None = None
) -> tuple[list[int], int, float]:
    """Prefill the prompt, then run the two-pass until a stop token / max_tokens.

    Args:
        runner: the live model runner to drive.
        lora_req: the Uno gated LoRA to use for the draft.
        prompt_ids: the (tokenized) prompt to decode from.
        args: parsed CLI args (block size, sampling, stop tokens, ...).
        max_steps: optional cap on the number of two-pass steps (for warmup).

    Returns:
        ``(tokens, num_steps, decode_seconds)`` for the generated span.
    """
    device = runner.device
    use_mask = args.noise_mode == "mask"
    # Random noise spans the full vocab; mask mode is a constant (the mask id).
    noise_max = args.mask_token_id if use_mask else runner.model_config.get_vocab_size()
    drv = PsiSpecRunner(
        runner,
        block_len=args.diffusion_block_size,
        mask_token_id=args.mask_token_id,
        temperature=args.temperature,
        noise_min=1,
        noise_max=noise_max,
        use_mask=use_mask,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    req = _seed_request(runner, lora_req, prompt_ids, args.max_tokens, args.temperature)
    try:
        p = len(prompt_ids)
        prompt_block = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_pos = torch.arange(p, dtype=torch.long, device=device).unsqueeze(0)
        # Prefill the prompt (LoRA off) to fill the KV cache.
        drv._forward_block(prompt_block, prompt_pos, torch.zeros(p, device=device))
        # Seed = last prompt token at position p-1; new tokens start at p.
        drv.set_request_state(0, frontier=p - 1, seed=prompt_ids[p - 1])

        stop = set(args.stop_token_ids or [])
        tokens: list[int] = []
        t0 = time.perf_counter()
        n_steps = 0
        while len(tokens) < args.max_tokens and (
            max_steps is None or n_steps < max_steps
        ):
            committed, lengths = drv.step()
            n_steps += 1
            new = [int(t) for t in committed[0, : int(lengths[0])].tolist()]
            tokens.extend(new)
            if stop and any(t in stop for t in new):
                break
        elapsed = time.perf_counter() - t0
    finally:
        _clear_request(runner, req.req_id)
    return tokens, n_steps, elapsed


def _measure_direct(
    llm, text: str, lora_req: LoRARequest | None, n_tokens: int, temperature: float
) -> dict:
    """Time prefill (pp) and per-decode (tg) for one config, and capture the text.

    Two calls split the phases: ``max_tokens=1`` is a cold prefill + 1 decode,
    ``max_tokens=n_tokens`` is a prefill + n decodes. With prefix caching off
    (see :func:`main`), differencing gives ``tg = (t_n - t_1) / (n - 1)`` and
    ``pp = t_1 - tg``. ``lora_req`` is the adapter for a direct-LoRA run or
    ``None`` for the plain base-AR baseline.
    """
    sp1 = SamplingParams(max_tokens=1, temperature=temperature)
    t0 = time.perf_counter()
    llm.generate(text, sp1, lora_request=lora_req, use_tqdm=False)
    t1 = time.perf_counter() - t0

    spn = SamplingParams(max_tokens=n_tokens, temperature=temperature)
    t0 = time.perf_counter()
    out = llm.generate(text, spn, lora_request=lora_req, use_tqdm=False)
    t_n = time.perf_counter() - t0

    tokens = out[0].outputs[0].token_ids
    n = len(tokens)
    tg = (t_n - t1) / (n - 1) if n > 1 else t1
    pp = t1 - tg
    return {
        "num_tokens": n,
        "text": out[0].outputs[0].text,
        "pp_ms": round(pp * 1000, 2),
        "tg_ms_per_token": round(tg * 1000, 2),
        "tg_tok_per_s": round(1000.0 / (tg * 1000), 2) if tg > 0 else None,
    }


def _warmup(llm, runner, lora_req: LoRARequest, tok, args) -> None:
    """Trigger the one-time Triton JIT compiles so the timed runs are steady-state.

    The two-pass (L-token attention + LoRA kernels) and the plain-AR path
    (1-token attention) compile different shapes, so both are warmed once.
    """
    text = tok.apply_chat_template(
        [{"role": "user", "content": "warmup"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    if args.mode == "direct":
        # Direct-LoRA (adapter on) and base (adapter off) 1-token decodes.
        llm.generate(
            text,
            SamplingParams(max_tokens=8, temperature=args.temperature),
            lora_request=lora_req,
            use_tqdm=False,
        )
        llm.generate(text, SamplingParams(max_tokens=8, temperature=args.temperature), use_tqdm=False)
        return
    ids = tok.encode(text)
    _two_pass_tokens(runner, lora_req, ids, args, max_steps=1)
    llm.generate(text, SamplingParams(max_tokens=8, temperature=args.temperature))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.tensor_parallel_size != 1:
        raise SystemExit(
            "the two-pass driver drives a single model runner; "
            "tensor-parallel > 1 is not wired yet (use --tensor-parallel-size 1)"
        )

    print("=== building in-process engine (V1 runner, LoRA on) ===", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        kv_cache_dtype=args.dtype,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=4,
        enable_lora=True,
        max_lora_rank=128,
        trust_remote_code=True,
        language_model_only=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Off so the direct-mode pp/tg differencing measures a real cold prefill
        # in both calls (the two-pass path manages its own KV and never uses it).
        enable_prefix_caching=False,
    )
    tok = llm.get_tokenizer()
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    lora_req = LoRARequest("uno", 1, args.gated_lora_path)
    runner.add_lora(lora_req)
    print("=== gated LoRA loaded ===", flush=True)

    if args.stop_token_ids is None:
        eos = getattr(tok, "eos_token_id", None)
        args.stop_token_ids = [eos] if eos is not None else []

    # Warm up the one-time Triton JIT compiles (two-pass + AR shapes) so the
    # timed runs below measure steady-state throughput, not first-use latency.
    _warmup(llm, runner, lora_req, tok, args)

    results = []
    for prompt in args.prompt:
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt_ids = tok.encode(text)

        if args.mode == "direct":
            # (1) Uno LoRA on the whole sequence; (2) plain base AR baseline.
            lora = _measure_direct(llm, text, lora_req, args.max_tokens, args.temperature)
            entry = {"prompt": prompt, "mode": "direct", "lora": lora}
            if args.compare_ar:
                base = _measure_direct(llm, text, None, args.max_tokens, args.temperature)
                entry["base_ar"] = base
                entry["correctness"] = {
                    "lora_differs_from_base": lora["text"] != base["text"],
                    "lora_tokens": lora["num_tokens"],
                    "base_tokens": base["num_tokens"],
                }
            results.append(entry)
            continue

        tokens, n_steps, elapsed = _two_pass_tokens(runner, lora_req, prompt_ids, args)
        entry = {
            "prompt": prompt,
            "text": tok.decode(tokens),
            "num_tokens": len(tokens),
            "num_steps": n_steps,
            "decode_seconds": round(elapsed, 4),
            "tokens_per_second": round(len(tokens) / elapsed, 2) if elapsed > 0 else None,
        }
        if args.compare_ar:
            t0 = time.perf_counter()
            base = llm.generate(
                text, SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
            )
            ar_elapsed = time.perf_counter() - t0
            ar_tokens = base[0].outputs[0].token_ids
            entry["ar_tokens"] = len(ar_tokens)
            entry["ar_seconds"] = round(ar_elapsed, 4)
            entry["ar_tokens_per_second"] = (
                round(len(ar_tokens) / ar_elapsed, 2) if ar_elapsed > 0 else None
            )
            if entry["tokens_per_second"] and entry["ar_tokens_per_second"]:
                entry["speedup_vs_ar"] = round(
                    entry["tokens_per_second"] / entry["ar_tokens_per_second"], 3
                )
        results.append(entry)

    print(
        json.dumps(
            {
                "model": args.model,
                "gated_lora_path": args.gated_lora_path,
                "generations": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
