# Pascal (sm_60/sm_61) Rebase Plan — v0.28.1rc1 → v0.29.0

**Date:** 2026-09-13
**Branch:** `pascal-sm61` on `/root/MyRepo/vllm-mainline`
**Base:** `c5d840ff6a` (merge base with origin/main)
**Target:** `origin/main` (v0.29.0, 621 commits ahead at time of analysis)
**Goal:** Rebase the 3 Pascal commits onto the latest mainline with minimal drift.

---

## 1. Current State

Three committed patches on `pascal-sm61` atop `c5d840ff6a`:

| Commit | Description | Files |
|--------|-------------|-------|
| `04f9bc1183` | feat: add Pascal compatibility support | P1–P7 + REBASE_PASCAL.md + pascal-patches/ |
| `5795c9db7c` | fix: complete Pascal build + runtime | dcp_direct_common.cuh + penalties.py |
| `6477b1ad0d` | fix: quant + sampling runtime fixes | auto_gptq.py, inc_wna16_linear.py, qwen3_5.py, qwen3_5_mtp.py, topk_topp_sampler.py |

Patch bundle in `pascal-patches/` (P1–P8) is the rebase source.

---

## 2. Upstream Changes Affecting Our Patches

Analyzed 621 commits on `origin/main` since `c5d840ff6a`. Six commits directly touch files we modified.

### 2.1 P3 — `vllm/v1/attention/ops/prefix_prefill.py` (LOW risk)

**Upstream commit:** `58dace61fa` — "prefix_prefill tiling independent of KV page size"

**What changed:** Removed the `is_pow2` branch that set `BLOCK_M/BLOCK_N` based on page size. Now unconditionally `BLOCK_M=128, BLOCK_N=64`.

**Impact on our patch:** Our P3 patch edits `BASE_BLOCK` (a different constant, used for tile alignment, not the BLOCK_M/N tiling). The two edits are in different code regions. **Zero merge conflict expected.**

**Action:** Re-apply P3 patch. Verify `BASE_BLOCK` still exists and our edit is intact.

---

### 2.2 P1 — `CMakeLists.txt` (LOW risk)

**Upstream changes:**
- Cutlass bumped from `v4.4.2` → `v4.7.1`
- Added `hisparse_kernels.cu` to source list
- Added `nvfp4_ds_mla_cache_kernels.cu`
- Removed `permute_cols.cu`
- Added `11.0f` to `FUSED_GDN_DECODE_ARCHS`
- ROCm Kimi-K3 / SM110 GDN additions

**Impact on our patch:** Our P1 adds `6.0;6.1` to two arch lists (≥12.8 and else). These lists have grown but our insertion points are still valid. **Zero merge conflict expected.**

**Action:** Re-apply P1 patch. Verify `6.0;6.1` entries are present in both arch lists.

---

### 2.3 P2 — `csrc/libtorch_stable/moe/moe_wna16.cu` (LOW risk)

**Upstream:** No changes since branch.

**Impact:** None. Our sm<700 half `atomicAdd` CAS shim is still needed.

**Action:** Re-apply P2 patch verbatim.

---

### 2.4 P4 — `csrc/libtorch_stable/quantization/gptq/q_gemm.cu` (MEDIUM risk)

**Upstream commit:** `2c9d68f80b` — "Remove GPTQ Group/Dynamic Activation Ordering" (535-line churn)

**What changed:**
- Removed `b_q_perm` parameter from all GPTQ kernel signatures (4-bit, 2-bit, 3-bit, 8-bit)
- Removed the `if (b_q_perm) a0 = a_ptr[b_q_perm[offset_k + t]]; else a0 = a_ptr[offset_k + t];` branching logic
- Kernel signature simplified: `const bool use_v2_format` only (no `const int* b_q_perm`)

**Impact on our patch:** Our P4 patch replaces `cublasHgemm` → `cublasGemmEx` with `CUDA_R_32F`. The diff context (line numbers around the `cublasHgemm` call) has shifted because upstream removed the `b_q_perm` parameter. **Context drift expected but the edit target (the `cublasHgemm` call site) is still present.**

**Action:**
1. Apply P4 patch with `git apply --3way` (prefer ours if conflict).
2. If conflict, manually locate the `cublasHgemm` call and re-apply the edit.
3. Verify the resulting function signature matches the new kernel signature (no `b_q_perm`).

---

### 2.5 P5/P6 — `vllm/v1/attention/backends/registry.py` + `flash_attn_legacy.py` (MEDIUM risk)

**Upstream changes:**
- `registry.py`: Added many new backend entries (FLASHMLA_SPARSE_DSV4, FLASHINFER_MLA_SPARSE_DSV4, ROCM_FLASHMLA_SPARSE_DSV4, MINIMAX_M3_SPARSE, CUTLASS_MSA, TRITON_MSA, HPC_ATTN, etc.)
- `flash_attn_legacy.py`: **Does NOT exist on origin/main** — it's our new file.

**Impact on P5:** Our P5 adds `FLASH_ATTN_LEGACY` to the enum. The enum has grown with many new entries. Our insertion point (after `FLASH_ATTN`) is still valid. **Zero merge conflict expected.**

**Impact on P6:** Our `flash_attn_legacy.py` is a new file. It will be preserved as-is during rebase since upstream has no competing file. However, the ABC it implements (`AttentionBackend`) has drifted:
- `supports_dcp` default changed from `True` → `False` in upstream (commit `db3814a4f2`)
- `naive_query_lens` property added to `CommonAttentionMetadata`
- `dcp_local_seq_lens_cpu` removed, replaced by `dcp_local_seq_lens_cpu_upper_bound`

**Action:**
1. Re-apply P5 patch (trivial enum addition).
2. Re-apply P6 patch (new file, no conflict).
3. **Post-rebase:** Audit `flash_attn_legacy.py` against the new ABC:
   - Set `supports_dcp: bool = False` (explicit, matching upstream default)
   - Add `naive_query_lens` property if required by the ABC
   - Remove any references to `dcp_local_seq_lens_cpu` (replaced by upper_bound)
   - Verify `get_supported_kernel_block_sizes()` is a `@staticmethod` (mainline convention)

---

### 2.6 P7 — `smoke_test_pascal.py` (NO risk)

**Upstream:** No changes. Self-contained test file.

**Action:** Re-apply P7 patch verbatim.

---

### 2.7 P8 — `csrc/libtorch_stable/attention/dcp_utils/dcp_direct_common.cuh` (NO risk)

**Upstream:** No changes since branch (commit `5795c9db7c`). +15 lines PTX guard for sm_70+ only.

**Impact:** None. Our Pascal build fix is still needed.

**Action:** Re-apply P8 patch verbatim.

---

## 3. Unstaged Runtime Fixes (from commit 6477b1ad0d)

These 5 files were committed locally but are NOT part of the patch bundle. They need re-evaluation against upstream.

**Also included:** `dcp_direct_common.cuh` from commit `5795c9db7c` (P8 in bundle, but also needs manual re-apply verification).

### 3.1 `auto_gptq.py` — removed `verify_marlin_supported()` hard-fail

**Upstream commit:** `2c9d68f80b` — "Remove GPTQ Group/Dynamic Activation Ordering"

**What changed upstream:**
- Added `normalize_and_validate_gptq_desc_act()` to `gptq_utils.py`
- Replaced inline `desc_act` normalization with call to new function
- Removed `RowvLLMParameter` import
- Removed `has_g_idx` from kernel config
- `marlin_repeat_scales_on_all_ranks()` signature changed: removed `act_order` parameter

**Impact on our fix:** Our fix (removing `verify_marlin_supported()` from `__init__`) is **still valid and needed**. The function still exists in upstream. However, the call site `marlin_repeat_scales_on_all_ranks(self.quant_config.desc_act, ...)` now needs updating — upstream removed the `act_order` parameter.

**Action:**
1. Re-apply our `verify_marlin_supported()` removal.
2. Update `marlin_repeat_scales_on_all_ranks()` call to drop the `desc_act` argument.
3. Verify `normalize_and_validate_gptq_desc_act` is imported (it's now in `gptq_utils.py`).

### 3.2 `inc_wna16_linear.py` — fallback to AutoGPTQ when marlin unavailable

**Upstream commit:** `2c9d68f80b`

**What changed upstream:**
- Renamed `self.inner_method` → `self.linear_method` throughout
- Added `assert isinstance(self.layer_config.group_size, int)` guards
- Removed `RowvLLMParameter` import

**Impact on our fix:** Our fallback logic (after `check_marlin_supported`, add `if not use_marlin and in_map: use_marlin = not current_platform.is_cpu() and not current_platform.is_xpu()`) is **still valid**. But the variable name changed from `inner_method` to `linear_method` — our edit references `self.inner_method` which no longer exists.

**Action:**
1. Re-apply our fallback logic.
2. Update all references from `self.inner_method` → `self.linear_method`.
3. Add the `assert isinstance(self.layer_config.group_size, int)` guards if not already present.

### 3.3 `qwen3_5.py` — pass `quant_config` to `VocabParallelEmbedding`

**Upstream:** No changes since branch.

**Action:** Re-apply verbatim.

### 3.4 `qwen3_5_mtp.py` — pass `quant_config` to `VocabParallelEmbedding`

**Upstream commit:** Changes to MTP pipeline parallelism support (`SupportsPP`, `intermediate_tensors` branching).

**What changed upstream:**
- Added `SupportsPP` mixin to `Qwen3_5MTP`
- Changed `is_first_rank` check to `intermediate_tensors is None` check
- Added `make_empty_intermediate_tensors` delegation

**Impact on our fix:** Our edit (passing `quant_config` to `VocabParallelEmbedding`) is in a different code region (the `__init__` method, embedding creation). The upstream changes are in the forward pass and class declaration. **Zero merge conflict expected.**

**Action:** Re-apply verbatim. Verify `VocabParallelEmbedding` call still exists with our edit.

### 3.5 `topk_topp_sampler.py` — wrap `FlashInferBackend` import in try/except

**Upstream commit:** `aed894c190` — "Migrate sampling, DFlash JIT kernels"

**What changed upstream:**
- Added `register_top_k_top_p_warmups()` function
- Restructured `apply_top_k_top_p()` to handle CPU path differently
- Added warmup registration in `TopKTopPSampler.__init__`

**Impact on our fix:** Our fix wraps the `FlashInferBackend` import in try/except. The upstream 37-line churn is concentrated in the `flashinfer_sample` region, adjacent to our `flashinfer_sampler_supported()` try/except. **Possible small re-target expected, not a guaranteed clean verbatim apply.**

**Action:** Re-apply with possible re-target. Verify the import location is still valid and the try/except still wraps the correct import.

### 3.6 `penalties.py` — torch fallback for sm<700 atomics

**Upstream:** No changes since branch.

**Action:** Re-apply verbatim.

---

## 4. Rebase Workflow

```bash
cd /root/MyRepo/vllm-mainline

# 1. Create rebase branch from latest mainline
git fetch origin main
git checkout -b pascal-sm61-rebase origin/main

# 2. Apply patch bundle (P1-P8)
git apply pascal-patches/P1-CMakeLists.txt.patch
git apply pascal-patches/P2-moe_wna16.cu.patch
git apply pascal-patches/P3-prefix_prefill.py.patch
git apply pascal-patches/P4-q_gemm.cu.patch
git apply pascal-patches/P5-registry.py.patch
git apply pascal-patches/P6-flash_attn_legacy.py.patch
git apply pascal-patches/P7-smoke_test_pascal.py.patch
git apply pascal-patches/P8-dcp_direct_common.cuh.patch

# 3. Apply unstaged runtime fixes (manually, since not in bundle)
#    Re-apply each fix from commit 6477b1ad0d, updating for upstream drift
#    Also verify P8 (dcp_direct_common.cuh) is intact

# 4. Resolve any conflicts
git diff --check
git status

# 5. Build C extension
source /root/.pyenv/versions/cuda12vllm/bin/activate
export PATH=/root/cuda_pascal.sh/bin:$PATH
VLLM_USE_PRECOMPILED=0 uv pip install -e . --torch-backend=auto

# 6. Run validation ladder (see Section 5)
```

---

## 5. Validation Ladder

| Level | Test | Command | Pass Criteria |
|-------|------|---------|---------------|
| L0 | Build | `source /root/.pyenv/versions/cuda12vllm/bin/activate && export PATH=/root/cuda_pascal.sh/bin:$PATH && pip install -e . --no-deps --no-build-isolation` (venv bin on PATH, setuptools>=77.0.3,<81.0.0, common.txt constraints) | Exit 0, no ptxas errors for sm_61 |
| L1 | Import + smoke | `python -c "import vllm; from vllm.v1.attention.backends.registry import AttentionBackendEnum; print(AttentionBackendEnum.FLASH_ATTN_LEGACY)"` | Prints `FLASH_ATTN_LEGACY` |
| L1b | Smoke test | `python smoke_test_pascal.py` | ALL PASSED |
| L2 | WNA16 GPTQ | `.venv/bin/python -m pytest tests/quantization/test_auto_round.py -v -k wna16` | 4/4 pass |
| L3 | Serve | Use known-good `/root/vllm_server_Q3.5.sh` config: `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, `--dtype float16`, `--tensor-parallel-size 2`, `--compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4]}'`, `--kv-cache-memory-bytes 3725884129`, `--language-model-only` (or `--skip-mm-profiling` for vision), `--max-model-len 40960` | Server up, `/health` 200, serves text (~5.75 GiB/GPU idle) |
| L4 | Regression | Full test suite for affected modules | No regressions |

---

## 6. Risk Ranking (peer assessment + my analysis)

| Rank | Patch | Risk | Reason |
|------|-------|------|--------|
| 1 | P6 (flash_attn_legacy.py) | **MEDIUM** | ABC drift: `supports_dcp` default changed, `naive_query_lens` added, `dcp_local_seq_lens_cpu` removed |
| 2 | P4 (q_gemm.cu) | **MEDIUM** | Kernel signature changed (b_q_perm removed), diff context drift |
| 3 | auto_gptq.py fix | **MEDIUM** | `marlin_repeat_scales_on_all_ranks` signature changed, `normalize_and_validate_gptq_desc_act` added |
| 4 | inc_wna16_linear.py fix | **MEDIUM** | `inner_method` → `linear_method` rename, assert guards added |
| 5 | P1 (CMakeLists.txt) | **LOW** | Arch lists grew but insertion points stable |
| 6 | P3 (prefix_prefill.py) | **LOW** | BASE_BLOCK edit is independent of upstream BLOCK_M/N change |
| 7 | P2 (moe_wna16.cu) | **LOW** | No upstream changes |
| 8 | P5 (registry.py) | **LOW** | Enum addition, insertion point stable |
| 9 | P7 (smoke_test_pascal.py) | **NO** | Self-contained, no upstream changes |
| 10 | P8 (dcp_direct_common.cuh) | **NO** | No upstream changes |
| 11 | qwen3_5.py/mtp.py fixes | **LOW** | Different code regions from upstream changes |
| 12 | penalties.py fix | **NO** | No upstream changes |
| 13 | topk_topp_sampler.py fix | **LOW** | Upstream churn adjacent to our edit; possible small re-target |

---

## 7. Post-Rebase Audit Checklist

- [ ] `flash_attn_legacy.py`: `supports_dcp = False` (explicit)
- [ ] `flash_attn_legacy.py`: `naive_query_lens` property present if ABC requires it
- [ ] `flash_attn_legacy.py`: No references to `dcp_local_seq_lens_cpu`
- [ ] `auto_gptq.py`: `marlin_repeat_scales_on_all_ranks()` call updated (no `act_order` arg)
- [ ] `inc_wna16_linear.py`: `self.inner_method` → `self.linear_method`
- [ ] `q_gemm.cu`: Kernel signatures match upstream (no `b_q_perm`)
- [ ] `CMakeLists.txt`: `6.0;6.1` present in both arch lists
- [ ] `prefix_prefill.py`: `BASE_BLOCK` edit intact
- [ ] `dcp_direct_common.cuh`: PTX guard intact (sm_70+ only)
- [ ] `topk_topp_sampler.py`: try/except still wraps correct import
- [ ] Build: C extension compiles for sm_61 without ptxas errors
- [ ] Smoke test: `smoke_test_pascal.py` ALL PASSED
- [ ] WNA16 GPTQ tests: 4/4 pass
- [ ] Serve: Qwen3.5-0.8B serves text correctly

---

## 8. Notes

- **Incremental rebase, NOT one git rebase:** Apply patches individually, verify each, then commit. This avoids losing work if a patch fails.
- **Do NOT use `requirements/cuda.txt`:** It pins torch 2.13/cu13 packages incompatible with sm_61. Use `requirements/common.txt` with constraints file.
- **setuptools pin:** Must be `>=77.0.3,<81.0.0`.
- **Venv bin on PATH:** Required for pip-provided cmake/ninja.
- **CWD shadowing:** Always verify `import vllm` from neutral CWD (`/root`, `/tmp`) or use `pip show vllm`.

---

## 9. Related Documentation

- `REBASE_PASCAL.md` — Original rebase documentation (pre-v0.29.0)
- `pascal-patches/` — Patch bundle (P1–P8)
- `MEMORY.md` — Session memory index
- `pascal-sm61-port-status.md` — Current port status
