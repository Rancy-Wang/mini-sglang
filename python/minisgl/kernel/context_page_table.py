from __future__ import annotations

from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


_CONTEXT_PAGE_TABLE_MODULE: Module | None = None
_CONTEXT_PAGE_TABLE_LOAD_ERROR: Exception | None = None
CONTEXT_PAGE_TABLE_BLOCK_SIZE = 256


def _load_context_page_table_module() -> Module:
    global _CONTEXT_PAGE_TABLE_LOAD_ERROR, _CONTEXT_PAGE_TABLE_MODULE
    if _CONTEXT_PAGE_TABLE_MODULE is not None:
        return _CONTEXT_PAGE_TABLE_MODULE
    if _CONTEXT_PAGE_TABLE_LOAD_ERROR is not None:
        raise RuntimeError(
            "The CUDA Context page-table compiler is unavailable."
        ) from _CONTEXT_PAGE_TABLE_LOAD_ERROR
    try:
        _CONTEXT_PAGE_TABLE_MODULE = load_aot(
            "context_page_table",
            cuda_files=["context_page_table.cu"],
        )
    except Exception as exc:
        _CONTEXT_PAGE_TABLE_LOAD_ERROR = exc
        raise
    return _CONTEXT_PAGE_TABLE_MODULE


def preload_context_page_table_kernel() -> None:
    """Compile/load the fixed CUDA extension before the server becomes ready."""

    _load_context_page_table_module()


def compile_context_page_table_aot(
    source: torch.Tensor,
    segment_table_indices: torch.Tensor | None,
    key_positions: torch.Tensor,
    key_offsets: torch.Tensor,
    *,
    max_seqlen_k: int,
    flat_indices: torch.Tensor | None,
    padded_page_table: torch.Tensor | None,
    direct: bool,
    block_size: int = CONTEXT_PAGE_TABLE_BLOCK_SIZE,
) -> None:
    """Launch one precompiled normal/direct and flat/padded page-table variant."""

    module = _load_context_page_table_module()
    if flat_indices is None and padded_page_table is None:
        raise ValueError("At least one Context page-table output must be requested.")

    if direct:
        if segment_table_indices is not None:
            raise ValueError("Direct Context pages do not use segment table indices.")
        if flat_indices is not None and padded_page_table is not None:
            module.compile_direct_page_table_both(
                source,
                key_positions,
                key_offsets,
                int(max_seqlen_k),
                int(block_size),
                flat_indices,
                padded_page_table,
            )
        elif flat_indices is not None:
            module.compile_direct_page_table_flat(
                source,
                key_positions,
                key_offsets,
                int(max_seqlen_k),
                int(block_size),
                flat_indices,
            )
        else:
            assert padded_page_table is not None
            module.compile_direct_page_table_padded(
                source,
                key_positions,
                key_offsets,
                int(max_seqlen_k),
                int(block_size),
                padded_page_table,
            )
        return

    if segment_table_indices is None:
        raise ValueError("Normal Context pages require segment table indices.")
    if flat_indices is not None and padded_page_table is not None:
        module.compile_context_page_table_both(
            source,
            segment_table_indices,
            key_positions,
            key_offsets,
            int(max_seqlen_k),
            int(block_size),
            flat_indices,
            padded_page_table,
        )
    elif flat_indices is not None:
        module.compile_context_page_table_flat(
            source,
            segment_table_indices,
            key_positions,
            key_offsets,
            int(max_seqlen_k),
            int(block_size),
            flat_indices,
        )
    else:
        assert padded_page_table is not None
        module.compile_context_page_table_padded(
            source,
            segment_table_indices,
            key_positions,
            key_offsets,
            int(max_seqlen_k),
            int(block_size),
            padded_page_table,
        )
