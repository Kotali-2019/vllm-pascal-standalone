"""Functional validation forcing the Pascal-specific FLASH_ATTN_LEGACY backend.

No monkeypatching: this exercises the real code path with
attention_config.backend = FLASH_ATTN_LEGACY, i.e. the fp16 flash-attn-legacy
C extension (flash_attn_legacy_cuda) running on the sm_61 GPU.

NOTE: the LLM construction + generate MUST live under an
`if __name__ == '__main__':` guard. vLLM's EngineCore spawns its subprocess
with the `spawn` start method, which re-imports this module; without the guard
the top-level LLM(...) would re-execute at import time and crash with
"An attempt has been made to start a new process before the current process
has finished its bootstrapping phase."

Run:
  source /root/cuda_pascal.sh >/dev/null 2>&1
  timeout 900 /root/vllm-mainline-test/bin/python _func_fa_legacy.py
"""
import vllm

assert "vllm-mainline" in vllm.__file__, f"WRONG vllm: {vllm.__file__}"
print(f"[setup] vllm = {vllm.__file__}", flush=True)

import torch  # noqa: E402

print(
    f"[setup] cuda 0 = {torch.cuda.get_device_name(0)} "
    f"(cc {torch.cuda.get_device_capability(0)})",
    flush=True,
)

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config.attention import AttentionConfig  # noqa: E402
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402

MODEL = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)


def main():
    print("[stage] LLM(...) with attention_config.backend=FLASH_ATTN_LEGACY",
          flush=True)
    llm = LLM(
        model=MODEL,
        dtype="half",
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        seed=0,
        kernel_config={"enable_jit_warmup": False, "enable_cutedsl_warmup": False},
        attention_config=AttentionConfig(
            backend=AttentionBackendEnum.FLASH_ATTN_LEGACY
        ),
    )
    print("[stage] LLM() returned — engine init SUCCEEDED", flush=True)

    outs = llm.generate(
        ["The capital of France is", "2 + 2 =",
         "The largest planet in the solar system is"],
        SamplingParams(max_tokens=32, temperature=0.0),
    )
    for o in outs:
        print(f"  prompt: {o.prompt!r}\n  output: {o.outputs[0].text!r}", flush=True)
    print("[done] FLASH_ATTN_LEGACY functional run COMPLETED", flush=True)


if __name__ == "__main__":
    main()
