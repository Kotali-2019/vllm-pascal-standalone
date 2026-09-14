# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend backed by the ``flash-attn-legacy`` (Pascal/Volta) kernel.

This is a *verification/debug* knob, not a performance path. The
``flash_attn_legacy_cuda.forward_varlen`` kernel is a correct-but-degenerate
FA2 kernel on sm_61 (Pascal) — it runs several times slower than the Triton
FA2 that ``TRITON_ATTN`` already uses. See the project memory for the A/B.

Two structural differences from the stock ``FLASH_ATTN`` backend drive the
design here:

* vLLM v1 keeps **paged** KV (``[2, num_blocks, block_size, H_kv, d]`` +
  ``block_table``), but the legacy kernel reads **contiguous packed** K/V
  (``[total_k, H_kv, d]`` + ``cu_seqlens``). So every forward de-pages the KV
  cache into a packed buffer first (``_depaginate``).
* The kernel exposes only a binary ``is_causal`` flag with an *aligned* causal
  mask (``k_idx <= q_idx``). That is only correct for (a) pure prefill
  (query is the whole sequence) or (b) single-token decode (query is the last
  token, causal off). Any chunked-prefill / mixed batch is not expressible, so
  those fall back to a (correct) torch reference (``_torch_fallback``).

Select it with ``VLLM_ATTENTION_BACKEND=FLASH_ATTN_LEGACY``.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm._custom_ops import reshape_and_cache_flash
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

# The legacy kernel is a separately-installed C extension. Import it lazily so
# that merely *having* this module present never breaks a run that uses a
# different backend; the ImportError is only surfaced if this backend is chosen.
try:
    import flash_attn_legacy_cuda as _fa_legacy_cuda
except ImportError as _err:  # pragma: no cover - depends on install
    _fa_legacy_cuda = None
    _FA_LEGACY_IMPORT_ERR = _err
else:
    _FA_LEGACY_IMPORT_ERR = None


@dataclass
class FlashAttentionLegacyMetadata:
    """Per-batch metadata for the legacy backend.

    GPU tensors are what the kernel / de-page consume; the ``*_cpu`` copies are
    used to make the (sync-free) dispatch decision in ``forward``.
    """

    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    total_seq_len: int
    query_start_loc: torch.Tensor  # [B + 1] GPU, int32
    seq_lens: torch.Tensor  # [B] GPU, int32 (computed + new, per request)
    block_table: torch.Tensor  # [B, max_blocks] GPU, int32
    slot_mapping: torch.Tensor
    # CPU mirrors for the dispatch decision (no GPU sync).
    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    num_computed_tokens_cpu: torch.Tensor


class FlashAttentionLegacyMetadataBuilder(
    AttentionMetadataBuilder[FlashAttentionLegacyMetadata]
):
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionLegacyMetadata:
        # Derive the CPU mirrors directly from the non-deprecated fields
        # (query_start_loc_cpu + seq_lens) rather than the deprecated
        # ``seq_lens_cpu`` / ``num_computed_tokens_cpu`` properties.
        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_cpu = common_attn_metadata.seq_lens.to("cpu")
        query_lens_cpu = qsl_cpu[1:] - qsl_cpu[:-1]
        num_computed_tokens_cpu = seq_lens_cpu - query_lens_cpu
        return FlashAttentionLegacyMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            total_seq_len=int(seq_lens_cpu.sum()),
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            query_start_loc_cpu=qsl_cpu,
            seq_lens_cpu=seq_lens_cpu,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
        )


class FlashAttentionLegacyImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        if _fa_legacy_cuda is None:
            raise ImportError(
                "flash_attn_legacy_cuda is not importable; the flash-attn-legacy "
                "wheel is not installed in this environment. Install it to use "
                "VLLM_ATTENTION_BACKEND=FLASH_ATTN_LEGACY."
            ) from _FA_LEGACY_IMPORT_ERR

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = (
            num_kv_heads if num_kv_heads is not None else num_heads
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionLegacyMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens == 0:
            return output.fill_(0)

        q = query[:num_actual_tokens]  # [total_q, Hq, d]

        key_cache, value_cache = kv_cache.unbind(0)

        # Write the current K/V into the paged cache (same op as FLASH_ATTN).
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        # De-page the paged KV into packed [total_k, Hkv, d] buffers.
        k_flat, v_flat = self._depaginate(key_cache, value_cache, attn_metadata)

        cu_q = attn_metadata.query_start_loc.to(torch.int32)
        # NOTE: torch.cumsum promotes int32 -> int64, so force int32 (the kernel
        # requires it) on the final concatenated tensor.
        cu_k = torch.cumsum(attn_metadata.seq_lens.to(torch.int32), 0)
        cu_k = torch.cat(
            (
                torch.zeros(1, device=cu_k.device, dtype=torch.int32),
                cu_k,
            )
        ).to(torch.int32)
        max_q = attn_metadata.max_query_len
        max_k = attn_metadata.max_seq_len

        # Dispatch on batch composition (sync-free: uses the *_cpu mirrors).
        q_lens_cpu = (
            attn_metadata.query_start_loc_cpu[1:]
            - attn_metadata.query_start_loc_cpu[:-1]
        )
        all_decode = bool((q_lens_cpu == 1).all())
        all_pure_prefill = bool((attn_metadata.num_computed_tokens_cpu == 0).all())

        if all_decode:
            # Single query per request attends to every past key.
            o = _fa_legacy_cuda.forward_varlen(
                q, k_flat, v_flat, cu_q, cu_k, max_q, max_k, self.scale, False
            )[0]
        elif all_pure_prefill:
            # Query is the whole (new) sequence; aligned causal mask is correct.
            o = _fa_legacy_cuda.forward_varlen(
                q, k_flat, v_flat, cu_q, cu_k, max_q, max_k, self.scale, True
            )[0]
        else:
            # Chunked prefill / mixed batch: not expressible with a single
            # is_causal flag, so use the (correct) torch reference.
            o = self._torch_fallback(
                q, k_flat, v_flat, cu_q, cu_k, attn_metadata
            )

        # ``output`` is ``empty_like(query)`` -> [num_actual_tokens, Hq, d]
        # (3D, same as ``o``), so copy directly (no 2D view).
        output[:num_actual_tokens].copy_(o)
        return output

    def _depaginate(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: FlashAttentionLegacyMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged KV into packed [total_k, Hkv, d] buffers.

        key_cache/value_cache: [num_blocks, block_size, Hkv, d]
        block_table:           [B, max_blocks] (int32, padded)
        seq_lens:              [B]  (total tokens per request)
        """
        block_table = attn_metadata.block_table  # [B, max_blocks]
        seq_lens = attn_metadata.seq_lens  # [B]
        B = seq_lens.shape[0]
        block_size = key_cache.shape[1]
        Hkv = key_cache.shape[2]
        d = key_cache.shape[3]

        # Fancy-index every referenced block: [B, max_blocks, block_size, Hkv, d]
        gk = key_cache[block_table]
        gv = value_cache[block_table]
        # Merge (max_blocks, block_size) into one capacity dim:
        #   [B, max_blocks * block_size, Hkv, d]  (contiguous -> free reshape)
        cap = block_table.shape[1] * block_size
        gk = gk.reshape(B, cap, Hkv, d)
        gv = gv.reshape(B, cap, Hkv, d)

        # Flat gather: row i -> (request, in-request offset).
        total_k = attn_metadata.total_seq_len
        dev = seq_lens.device
        idx_dtype = seq_lens.dtype
        # start[i] = sum(seq_lens[:i]) = where request i begins in the packed buf.
        # (cumsum has length B; prepend a 0 to get the B start offsets.)
        start = torch.cat(
            (torch.zeros(1, device=dev, dtype=idx_dtype), torch.cumsum(seq_lens, 0)[:-1])
        )
        row = torch.repeat_interleave(
            torch.arange(B, device=dev, dtype=idx_dtype), seq_lens
        )
        col = torch.arange(total_k, device=dev, dtype=idx_dtype) - start[row]
        k_flat = gk[row, col]  # [total_k, Hkv, d]
        v_flat = gv[row, col]
        return k_flat, v_flat

    def _torch_fallback(
        self,
        q: torch.Tensor,
        k_flat: torch.Tensor,
        v_flat: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        attn_metadata: FlashAttentionLegacyMetadata,
    ) -> torch.Tensor:
        """Correct (but slow) reference for the non-expressible cases.

        Handles chunked prefill (query is a suffix of the sequence) and mixed
        batches. Per request: query token at absolute position ``p`` attends to
        keys with absolute position ``<= p``.
        """
        Hq = q.shape[1]
        Hkv = k_flat.shape[1]
        rep = Hq // Hkv
        out = torch.empty_like(q)

        cu_q_cpu = attn_metadata.query_start_loc_cpu
        cu_k_cpu = torch.cumsum(attn_metadata.seq_lens_cpu, 0)
        num_computed_cpu = attn_metadata.num_computed_tokens_cpu
        dev = q.device

        for i in range(cu_q_cpu.shape[0] - 1):
            q0, q1 = int(cu_q_cpu[i]), int(cu_q_cpu[i + 1])
            k0, k1 = int(cu_k_cpu[i - 1]) if i > 0 else 0, int(cu_k_cpu[i])
            qi = q[q0:q1]  # [ql, Hq, d]
            ki = k_flat[k0:k1]  # [sl, Hkv, d]
            vi = v_flat[k0:k1]

            ke = ki.repeat_interleave(rep, dim=1)  # [sl, Hq, d]
            ve = vi.repeat_interleave(rep, dim=1)
            q3 = qi.float().transpose(0, 1)  # [Hq, ql, d]
            k3 = ke.float().transpose(0, 1)  # [Hq, sl, d]
            v3 = ve.float().transpose(0, 1)
            s = torch.bmm(q3, k3.transpose(1, 2)) * self.scale  # [Hq, ql, sl]

            ql = q3.shape[1]
            nc = int(num_computed_cpu[i])
            qabs = nc + torch.arange(ql, device=dev)
            kabs = torch.arange(ki.shape[0], device=dev)
            mask = kabs[None, :] <= qabs[:, None]  # [ql, sl]
            s = s.masked_fill(~mask.unsqueeze(0), float("-inf"))
            p = torch.softmax(s, dim=-1)
            o = torch.bmm(p, v3)  # [Hq, ql, d]
            out[q0:q1] = o.transpose(0, 1).to(q.dtype)
        return out


class FlashAttentionLegacyBackend(AttentionBackend):
    accept_output_buffer: bool = True
    # The legacy kernel is fp16-only.
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]
    # forward() writes the current K/V into the paged cache (reshape_and_cache),
    # so the framework must not issue a separate KV-cache update.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # De-paging works with any block size.
        return [MultipleOf(1)]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_LEGACY"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        return FlashAttentionLegacyImpl

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        return FlashAttentionLegacyMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Kernel asserts head_dim in {32, 64, 96, 128, 256}.
        return head_size in (32, 64, 96, 128, 256)

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype == torch.float16

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # The kernel targets Pascal (6.x) and Volta (7.x).
        return capability.major in (6, 7)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER
