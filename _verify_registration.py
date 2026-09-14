"""Verify FLASH_ATTN_LEGACY (and TRITON_ATTN) are registered + importable at
runtime, from the mainline vllm tree. Run with the mainline test python."""
import vllm

assert "vllm-mainline" in vllm.__file__, f"WRONG vllm: {vllm.__file__}"
print(f"[ok] vllm = {vllm.__file__}", flush=True)

from vllm.v1.attention.backends.registry import AttentionBackendEnum as B  # noqa: E402

# 1) the enum member exists (this is what VLLM_ATTENTION_BACKEND resolves to)
print(f"[ok] 'FLASH_ATTN_LEGACY' in AttentionBackendEnum.__members__ = "
      f"{'FLASH_ATTN_LEGACY' in B.__members__}", flush=True)

# 2) resolve the class for each Pascal-relevant backend (imports the module)
for name in ("FLASH_ATTN_LEGACY", "TRITON_ATTN"):
    member = B[name]  # uses the metaclass __getitem__
    cls = member.get_class()
    print(f"[ok] {name:16s} -> {cls.__module__}.{cls.__name__}", flush=True)

print("[done] registration check PASSED", flush=True)
