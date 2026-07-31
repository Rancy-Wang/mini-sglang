from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.utils import is_sm100_supported

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class FACaptureData(BaseCaptureData):
    pass


@dataclass
class FAMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor
    context_mask_aux: tuple[torch.Tensor, torch.Tensor] | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


def is_fa_context_mask_supported(device: torch.device | int | None = None) -> bool:
    """Whether FA4's CuTe custom mask can run on the selected CUDA device."""

    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major in (9, 10, 11)


def validate_fa_context_mask_support(device: torch.device | int | None = None) -> None:
    if not is_fa_context_mask_supported(device):
        capability = torch.cuda.get_device_capability(device) if torch.cuda.is_available() else None
        raise ValueError(
            "--contextual-prefill-mode flashattention-mask requires the FA4 CuTe "
            "custom-mask kernel on SM90/SM100/SM110; "
            f"current CUDA capability is {capability}. Use flashinfer-mask on SM80."
        )

    try:
        import inspect

        from sgl_kernel import _fa4_interface
    except Exception as exc:
        raise RuntimeError(
            "FlashAttention context-mask mode requires a working sgl-kernel FA4 "
            "installation with CuTe DSL support."
        ) from exc

    public_has_mask = (
        "mask_mod" in inspect.signature(_fa4_interface.flash_attn_varlen_func).parameters
    )
    private_has_mask = hasattr(_fa4_interface, "_flash_attn_fwd") and (
        "mask_mod" in inspect.signature(_fa4_interface._flash_attn_fwd).parameters
    )
    if not (public_has_mask or private_has_mask):
        raise RuntimeError(
            "The installed sgl-kernel FA4 interface does not expose mask_mod. "
            "Install a build with FA4 custom-mask support."
        )


@lru_cache(maxsize=1)
def _get_context_visibility_mask_mod():
    import cutlass.cute as cute

    @cute.jit
    def context_visibility_mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        full_token_visible_until, query_start = aux_tensors
        query_position = q_idx + query_start[0]
        return (kv_idx <= query_position) & (
            query_position < full_token_visible_until[kv_idx]
        )

    return context_visibility_mask_mod


class FlashAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.capture: FACaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim**-0.5
        self.version = 4 if is_sm100_supported() else 3

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return _fa_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            version=self.version,
            context_mask_aux=metadata.context_mask_aux,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        masked_reqs = [req for req in reqs if req.use_context_mask]
        if masked_reqs:
            if not batch.is_prefill or batch.size != 1 or len(reqs) != 1:
                raise RuntimeError(
                    "FlashAttention context-mask Prefill must be an unpadded, single-request batch."
                )
            if len(masked_reqs) != 1:
                raise RuntimeError(
                    "Mixed masked and ordinary FlashAttention requests are unsupported."
                )

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        page_table = get_global_ctx().page_table
        new_page_table = torch.stack(
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")

        context_mask_aux = None
        if masked_reqs:
            if self.page_size != 1:
                raise RuntimeError(
                    "FlashAttention context-mask Prefill currently requires page_size=1."
                )
            req = masked_reqs[0]
            if req.full_token_visible_until is None:
                raise RuntimeError("Context-mask Prefill request is missing visibility metadata.")
            context_mask_aux = (
                req.full_token_visible_until.to(
                    device=device, dtype=torch.int32, non_blocking=True
                ),
                torch.tensor([req.cached_len], device=device, dtype=torch.int32),
            )
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
            context_mask_aux=context_mask_aux,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FACaptureData.create(max_bs, max_seq_len // self.page_size, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FAMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)


@lru_cache(maxsize=1)
def _get_fa4_context_mask_runner():
    import inspect

    try:
        from sgl_kernel import _fa4_interface
    except Exception as exc:
        raise RuntimeError(
            "Unable to import the sgl-kernel FA4 CuTe custom-mask interface."
        ) from exc

    public_func = _fa4_interface.flash_attn_varlen_func
    if "mask_mod" in inspect.signature(public_func).parameters:
        return public_func, True

    private_func = getattr(_fa4_interface, "_flash_attn_fwd", None)
    if private_func is None or "mask_mod" not in inspect.signature(private_func).parameters:
        raise RuntimeError("The installed sgl-kernel FA4 interface does not provide mask_mod.")
    return private_func, False


def _fa4_context_mask_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_mask_aux: tuple[torch.Tensor, torch.Tensor],
    softmax_scale: float,
    softcap: float,
    pack_gqa: bool | None,
) -> torch.Tensor:
    if q.ndim != 3 or page_table.ndim != 2 or page_table.size(0) != 1:
        raise RuntimeError("FA4 context-mask adapter expects flattened Q and one page-table row.")
    if k_cache.ndim != 4 or k_cache.size(1) != 1 or v_cache.shape != k_cache.shape:
        raise RuntimeError("FA4 context-mask adapter requires page_size=1 MHA KV cache.")

    # FA4 on Hopper does not accept this engine's page_size=1 MHA page table.
    # Gather the logical sequence on GPU; the CuTe attention/mask kernel then sees
    # fixed-shape [batch=1, sequence, heads, dim] tensors with no varlen metadata.
    page_indices = page_table[0].to(dtype=torch.int64)
    dense_k = k_cache.index_select(0, page_indices).squeeze(1).unsqueeze(0)
    dense_v = v_cache.index_select(0, page_indices).squeeze(1).unsqueeze(0)
    q_batched = q.unsqueeze(0)

    runner, is_public = _get_fa4_context_mask_runner()
    kwargs = {
        "q": q_batched,
        "k": dense_k,
        "v": dense_v,
        "softmax_scale": softmax_scale,
        "causal": False,
        "softcap": softcap,
        "num_splits": 1,
        "pack_gqa": pack_gqa,
        "mask_mod": _get_context_visibility_mask_mod(),
        "aux_tensors": list(context_mask_aux),
    }
    if is_public:
        kwargs["window_size"] = (None, None)
        out = runner(**kwargs)
    else:
        kwargs.update(
            window_size_left=None,
            window_size_right=None,
            return_lse=False,
        )
        out, _ = runner(**kwargs)
    return out.squeeze(0)


def _fa_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    version: int,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa: bool | None = None,  # Can be tuned for speed
    causal: bool = True,
    context_mask_aux: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if context_mask_aux is not None:
        return _fa4_context_mask_impl(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            context_mask_aux=context_mask_aux,
            softmax_scale=softmax_scale,
            softcap=softcap,
            pack_gqa=pack_gqa,
        )

    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
    except ImportError as e:
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`.\n"
            "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
        ) from e

    return flash_attn_with_kvcache(  # type: ignore
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        sm_margin=sm_margin,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        causal=causal,
        ver=version,  # TODO: support FA4 on blackwell
    )
