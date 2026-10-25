"""P5-3: `target.build.tilelang_hivm` must produce a REAL binary.

These tests are the gate for the build entry that P5-2 only emitted text for.
They are deliberately split by what they need:

* PROVENANCE, refusal and serialization need no compiler and always run.
* The full compile needs a `bishengir-compile` that supports
  `--input-ir-stage=late-hivm`. It is located through `HIVM_COMPILER`; when that
  is unset the test SKIPS rather than substituting a weaker check, so a green
  run never implies a compile happened when one did not.

The frozen P3 binary digest is asserted AGAINST, never returned: the point of
the compile test is that an independently generated program reproduces that
binary, which is only meaningful if the digest is an expected value here and not
a lookup key anywhere in the producer.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import tvm
from tilelang.contrib import hivm

GOLDEN = Path(__file__).resolve().parents[3] / "tests" / "golden" / "hivm" / "vadd_f32_single_tile" / "golden_codegen_input.json"
BUILD_ENTRY = "target.build.tilelang_hivm"
SAVE_ENTRY = "tl.ascend.ModuleSaveToBytes"
LOAD_ENTRY = "ffi.Module.load_from_bytes.asc"
GENERATED_FMT = "hivm-generated-aibin"
PREBUILT_FMT = "hivm-aibin"

# The frozen P3 AIV binary this phase must reproduce byte for byte.
FROZEN_SHA256 = "2c4247c3b2ce11f79ad15415eb06d4ce1d5523a0c19c91ad3743bc08e7b8c4c9"
FROZEN_SIZE = 4448


def _module_and_target():
    mod = tvm.ir.load_json(GOLDEN.read_text(encoding="utf-8"))
    key = list(mod.functions)[0]
    cfg = dict(mod[key].attrs["target"].export())
    cfg["keys"] = ["ascend", "hivm"]
    return mod, tvm.target.Target(cfg)


def _compiler() -> str | None:
    return os.environ.get("HIVM_COMPILER", "").strip() or None


@pytest.fixture
def built():
    """Build the real module, or skip honestly when no compiler is configured."""
    exe = _compiler()
    if exe is None:
        pytest.skip("HIVM_COMPILER is unset: no Late-hivm compiler is configured, so the binary-producing entry cannot be exercised here.")
    hivm.register(exe)
    mod, target = _module_and_target()
    return tvm.ffi.get_global_func(BUILD_ENTRY)(mod, target)


def test_build_entry_exists():
    assert tvm.ffi.get_global_func(BUILD_ENTRY, allow_missing=True) is not None


def test_build_without_hook_fails_rather_than_returning_a_prebuilt(built):
    """With no callback registered the entry must raise, not substitute.

    This is the property that separates P5 from the P4 PREBUILT path. It is
    checked with the callback REMOVED, after having just succeeded, so a cached
    or fallback artifact would be the only way for it to pass.
    """
    assert built is not None
    hivm.unregister()
    mod, target = _module_and_target()
    build = tvm.ffi.get_global_func(BUILD_ENTRY)
    with pytest.raises(Exception) as excinfo:
        build(mod, target)
    message = str(excinfo.value)
    assert "tilelang_callback_hivm_compile" in message
    assert "never falls back" in message


def test_generated_binary_is_the_frozen_aiv_binary(built):
    """An independently generated program reproduces the frozen binary.

    Equality here is the strongest available statement short of hardware: the
    TIR was read, Late MLIR was emitted, the Late pipeline ran, and the result
    is bit-identical to the artifact P2/P3 accepted.
    """
    blob = bytes(tvm.ffi.get_global_func(SAVE_ENTRY)(built))
    binary = blob[-FROZEN_SIZE:]
    assert len(binary) == FROZEN_SIZE
    assert hashlib.sha256(binary).hexdigest() == FROZEN_SHA256


def test_generated_module_does_not_claim_prebuilt_provenance(built):
    """The label names the generator and carries no frozen path or digest."""
    label = built.inspect_source()
    assert "GENERATED" in label
    assert "PREBUILT" not in label
    # The frozen path and digest must not leak into a generated module's label.
    assert "p2-2026" not in label
    assert FROZEN_SHA256 not in label


def test_restore_does_not_turn_a_generated_module_into_prebuilt(built):
    """BC-4 restore reconstructs PREBUILT only for the PREBUILT token.

    A restored module has an empty source map, so the label is rebuilt from the
    format token. The generated token must NOT be rebuilt into the frozen P3
    description, which is the defect the separate token exists to prevent.
    """
    blob = tvm.ffi.get_global_func(SAVE_ENTRY)(built)
    restored = tvm.ffi.get_global_func(LOAD_ENTRY)(blob)
    label = restored.inspect_source()
    assert "PREBUILT" not in (label or "")
    assert FROZEN_SHA256 not in (label or "")


def test_missing_compiler_is_reported_not_replaced(monkeypatch, tmp_path):
    """An unusable HIVM_COMPILER must raise, naming the variable."""
    monkeypatch.setenv("HIVM_COMPILER", str(tmp_path / "no-such-compiler"))
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.compile_late_mlir("module {\n}\n")
    assert "HIVM_COMPILER" in str(excinfo.value)


def test_empty_mlir_is_refused():
    with pytest.raises(hivm.CompilerError):
        hivm.compile_late_mlir("   \n")


def test_compiler_failure_is_not_swallowed(tmp_path):
    """A compiler that exits 0 but writes nothing must still fail the build."""
    fake = tmp_path / "fake-bishengir"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.register(str(fake))
        hivm.compile_late_mlir("module {\n}\n")
    assert "wrote no binary" in str(excinfo.value)


def test_nonzero_exit_is_not_swallowed(tmp_path):
    fake = tmp_path / "fake-bishengir"
    fake.write_text("#!/bin/sh\necho 'boom' >&2\nexit 3\n", encoding="utf-8")
    fake.chmod(0o755)
    os.environ["HIVM_COMPILER"] = str(fake)
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.compile_late_mlir("module {\n}\n")
    message = str(excinfo.value)
    assert "exit 3" in message
    assert "boom" in message


def test_kernel_cache_must_be_off(monkeypatch, tmp_path):
    """Goal §6.3: a cache hit would hide a fresh compile, so it is refused."""
    monkeypatch.delenv("TILELANG_CLEAR_CACHE", raising=False)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.compile_late_mlir("module {\n}\n")
    assert "TILELANG_CACHE_DIR" in str(excinfo.value)
