from pathlib import Path

import pytest

from minisgl.kernel import pynccl


def test_find_nccl_library_prefers_explicit_override(tmp_path, monkeypatch):
    library = tmp_path / "libnccl-custom.so.2"
    library.touch()
    monkeypatch.setenv("MINISGL_NCCL_SO_PATH", str(library))
    monkeypatch.setattr(pynccl, "_nccl_library_directories", lambda: [])

    assert pynccl._find_nccl_library() == library.resolve()


def test_find_nccl_library_rejects_invalid_override(tmp_path, monkeypatch):
    missing = tmp_path / "missing-libnccl.so.2"
    monkeypatch.setenv("MINISGL_NCCL_SO_PATH", str(missing))

    with pytest.raises(RuntimeError, match="MINISGL_NCCL_SO_PATH must point"):
        pynccl._find_nccl_library()


def test_find_nccl_library_accepts_versioned_package_library(tmp_path, monkeypatch):
    library_dir = tmp_path / "nvidia" / "nccl" / "lib"
    library_dir.mkdir(parents=True)
    library = library_dir / "libnccl.so.2"
    library.touch()
    monkeypatch.delenv("MINISGL_NCCL_SO_PATH", raising=False)
    monkeypatch.setattr(pynccl, "_nccl_library_directories", lambda: [library_dir])

    assert pynccl._find_nccl_library() == library.resolve()


def test_load_nccl_module_links_concrete_library_with_rpath(tmp_path, monkeypatch):
    library_dir = tmp_path / "nccl" / "lib"
    library_dir.mkdir(parents=True)
    library = library_dir / "libnccl.so.2"
    library.touch()
    captured = {}

    def fake_load_aot(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(pynccl, "_find_nccl_library", lambda: library)
    monkeypatch.setattr(pynccl, "load_aot", fake_load_aot)
    pynccl._load_nccl_module.cache_clear()
    try:
        pynccl._load_nccl_module()
    finally:
        pynccl._load_nccl_module.cache_clear()

    assert captured["args"] == ("pynccl",)
    assert captured["kwargs"]["cuda_files"] == ["pynccl.cu"]
    assert captured["kwargs"]["extra_ldflags"] == [
        str(library),
        f"-Wl,-rpath,{library_dir}",
    ]
    assert "-lnccl" not in captured["kwargs"]["extra_ldflags"]


def test_find_nccl_library_reports_searched_locations(tmp_path, monkeypatch):
    monkeypatch.delenv("MINISGL_NCCL_SO_PATH", raising=False)
    monkeypatch.setattr(pynccl, "_nccl_library_directories", lambda: [tmp_path])

    with pytest.raises(RuntimeError, match=str(Path(tmp_path) / "libnccl.so.2")):
        pynccl._find_nccl_library()
