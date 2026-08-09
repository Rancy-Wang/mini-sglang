from __future__ import annotations

import functools
import importlib.metadata
import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from minisgl.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


_NCCL_LIBRARY_ENV = "MINISGL_NCCL_SO_PATH"
_NCCL_LIBRARY_NAMES = ("libnccl.so.2", "libnccl.so")


def _nccl_library_directories() -> list[Path]:
    directories: list[Path] = []

    try:
        distribution = importlib.metadata.distribution("nvidia-nccl-cu12")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        directories.append(Path(distribution.locate_file("nvidia/nccl/lib")))

    try:
        spec = importlib.util.find_spec("nvidia.nccl")
    except ModuleNotFoundError:
        spec = None
    if spec is not None and spec.submodule_search_locations is not None:
        directories.extend(Path(location) / "lib" for location in spec.submodule_search_locations)

    for path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if path:
            directories.append(Path(path))

    # Preserve the discovery order while avoiding repeated checks and noisy errors.
    return list(dict.fromkeys(directories))


def _find_nccl_library() -> Path:
    override = os.environ.get(_NCCL_LIBRARY_ENV)
    if override:
        override_path = Path(override).expanduser()
        if not override_path.is_file():
            raise RuntimeError(
                f"{_NCCL_LIBRARY_ENV} must point to an existing NCCL shared library, "
                f"but got: {override_path}"
            )
        return override_path.resolve()

    searched: list[Path] = []
    for directory in _nccl_library_directories():
        for name in _NCCL_LIBRARY_NAMES:
            candidate = directory / name
            searched.append(candidate)
            if candidate.is_file():
                return candidate.resolve()

        # Some installations only retain the fully versioned file.
        for candidate in sorted(directory.glob("libnccl.so.2.*"), reverse=True):
            searched.append(candidate)
            if candidate.is_file():
                return candidate.resolve()

    searched_text = ", ".join(str(path) for path in searched) or "no candidate directories"
    raise RuntimeError(
        "Unable to locate an NCCL shared library for the PyNCCL AOT module. "
        "Install nvidia-nccl-cu12 or set MINISGL_NCCL_SO_PATH to libnccl.so.2. "
        f"Searched: {searched_text}"
    )


@functools.cache
def _load_nccl_module() -> Module:
    nccl_library = _find_nccl_library()
    return load_aot(
        "pynccl",
        cuda_files=["pynccl.cu"],
        extra_ldflags=[
            str(nccl_library),
            f"-Wl,-rpath,{nccl_library.parent}",
        ],
    )


@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
