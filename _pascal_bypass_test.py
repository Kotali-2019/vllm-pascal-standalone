"""
Pascal (sm_61) runtime-blocker proof — causal experiment.

Question under test: is the _bincount_kernel ptxas failure (the ONLY sm_70+
PTX in the Qwen3-0.6B path) the sole thing stopping engine init? If we remove
just that kernel launch, does init proceed further (and ideally complete +
generate)?

Three modes, selected by $PASCAL_MODE:
  control   - no patch. Expected to reproduce the exact Run-3 crash
               (penalties.py -> bincount -> _bincount_kernel -> ptxas .acq_rel).
  bypass    - monkeypatch penalties.bincount to a no-op. Isolates the failing
               kernel; everything else in the warmup/forward runs unmodified.
  nowarmup  - monkeypatch SamplingParams.for_sampler_warmup to a neutral
               (penalty-free) SamplingParams. Uses vLLM's OWN guard
               (`if self._new_penalties_reqs:`) to skip the kernel with no
               bypass of any code.

Run (from /root/MyRepo/vllm-mainline so this dir is sys.path[0] -> mainline vllm):
  source /root/cuda_pascal.sh >/dev/null 2>&1
  PASCAL_MODE=bypass timeout 600 /root/vllm-mainline-test/bin/python \
      /root/MyRepo/vllm-mainline/_pascal_bypass_test.py
"""
import os
import sys
import traceback

# --- sanity: we must be importing the MAINLINE tree, not the fork ---
import vllm  # noqa: E402

assert "vllm-mainline" in vllm.__file__, f"WRONG vllm: {vllm.__file__}"
print(f"[setup] vllm = {vllm.__file__}", flush=True)

MODE = os.environ.get("PASCAL_MODE", "control")
print(f"[setup] PASCAL_MODE = {MODE}", flush=True)

# --- apply the mode-specific patch (BEFORE any engine is constructed) ---
if MODE == "bypass":
    import vllm.v1.worker.gpu.sample.penalties as _pen

    _orig = _pen.bincount

    def _noop_bincount(*a, **k):
        return None

    _pen.bincount = _noop_bincount
    print("[patch] penalties.bincount -> no-op (kernel launch skipped)", flush=True)

elif MODE == "nowarmup":
    from vllm.sampling_params import SamplingParams

    def _neutral_warmup() -> "SamplingParams":
        # penalty-free, no logit_bias / bad_words / logprobs -> use_penalty() is
        # False, so add_request never populates _new_penalties_reqs and
        # apply_staged_writes skips bincount via vLLM's own guard.
        return SamplingParams(temperature=0.0)

    SamplingParams.for_sampler_warmup = staticmethod(_neutral_warmup)
    print("[patch] SamplingParams.for_sampler_warmup -> neutral (no penalties)", flush=True)

elif MODE == "control":
    print("[patch] none (control — expect the Run-3 ptxas crash)", flush=True)
else:
    raise SystemExit(f"unknown PASCAL_MODE {MODE!r}")

# --- device check ---
import torch  # noqa: E402

print(f"[setup] cuda device 0 = {torch.cuda.get_device_name(0)} "
      f"(cc {torch.cuda.get_device_capability(0)})", flush=True)

MODEL = ("/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
         "snapshots/c1899de289a04d12100db370d81485cdf75e47ca")


def main():
    from vllm import LLM, SamplingParams

    print("[stage] constructing LLM(...) — engine init begins", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="half",
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        seed=0,
        kernel_config={"enable_jit_warmup": False, "enable_cutedsl_warmup": False},
    )
    print("[stage] LLM() returned — engine init SUCCEEDED", flush=True)

    print("[stage] running generate(...)", flush=True)
    outs = llm.generate(
        ["The capital of France is", "2 + 2 =",
         "The largest planet in the solar system is"],
        SamplingParams(max_tokens=32, temperature=0.0),
    )
    print("[stage] generate() returned", flush=True)
    for o in outs:
        print(f"  prompt: {o.prompt!r}\n  output: {o.outputs[0].text!r}", flush=True)
    print(f"[done] MODE={MODE} COMPLETED — engine init + generate OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print(f"[FAIL] MODE={MODE} raised (see traceback):", flush=True)
        traceback.print_exc()
        # Distinguish "crashed at the bincount ptxas step" from "got further".
        tb = traceback.format_exc()
        if "_bincount_kernel" in tb or "Feature '.acq_rel'" in tb:
            print("[diagnosis] still failing at _bincount_kernel (the sm_70+ atomic)", flush=True)
        else:
            print("[diagnosis] failed at a DIFFERENT point than _bincount_kernel", flush=True)
        sys.exit(3)
