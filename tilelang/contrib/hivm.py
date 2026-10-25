"""Compiler callback for the TileLang HIVM DeviceCodeGen (P5-3).

The C++ build entry `target.build.tilelang_hivm` emits Late HIVM/AVE MLIR from
the prepared TIR and then looks up a registered function named
`tilelang_callback_hivm_compile`, handing it the MLIR text. This module is the
registration and the implementation:

    prepared TIR -> (C++ visitor) -> Late HIVM/AVE MLIR -> bishengir-compile
                 -> Ascend950PR AIV binary

Two properties of this file matter more than the code:

* It NEVER falls back to the frozen PREBUILT binary. If the compiler is
  missing, exits non-zero, or writes no output, the error is raised and the
  build fails. Returning a previously built artifact instead is exactly the
  substitution P5 exists to remove.
* It carries no digest of any known artefact. The only inputs are the MLIR text
  it is handed and the compiler it is told to run.

The compiler location is configuration, not a constant, because the AscendNPU-IR
build that provides it lives outside this repository. `HIVM_COMPILER` names the
executable; there is no default that silently picks a different one.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import tvm.ffi as tvm_ffi

__all__ = ["register", "unregister", "compile_late_mlir", "CompilerError"]

_HOOK = "tilelang_callback_hivm_compile"
_ENV_COMPILER = "HIVM_COMPILER"
_ENV_TARGET = "HIVM_TARGET"
_DEFAULT_TARGET = "Ascend950PR_9579"
_TIMEOUT_S = 1800


class CompilerError(RuntimeError):
    """The Late pipeline did not produce a binary.

    Raised instead of falling back to any prebuilt artifact.
    """


def _kernel_cache_disabled() -> None:
    """Fail if the kernel cache would hide a fresh compile (goal §6.3).

    P5-3 requires that every build in this phase actually invokes the compiler.
    A cache hit would return an artifact whose provenance this phase has not
    proven, so it is refused rather than tolerated.
    """
    if os.environ.get("TILELANG_CLEAR_CACHE", "").strip() in ("1", "true", "True"):
        return
    # The cache only participates when the kernel dir is set; an unset dir means
    # every compile is fresh, which is the state this phase needs.
    if not os.environ.get("TILELANG_CACHE_DIR", "").strip():
        return
    raise CompilerError(
        "HIVM CODEGEN: TILELANG_CACHE_DIR is set, so a cached artifact could "
        "be returned instead of a fresh compile. Unset it, or set "
        "TILELANG_CLEAR_CACHE=1, before building with target.build.tilelang_hivm."
    )


def _resolve_compiler() -> str:
    exe = os.environ.get(_ENV_COMPILER, "").strip()
    if not exe:
        raise CompilerError(
            f"HIVM CODEGEN: {_ENV_COMPILER} is not set, so the generated Late "
            "MLIR cannot be compiled. Point it at a bishengir-compile that "
            "supports `--input-ir-stage=late-hivm`. This callback never falls "
            "back to the frozen PREBUILT binary."
        )
    found = shutil.which(exe) or (exe if Path(exe).is_file() else None)
    if not found:
        raise CompilerError(f"HIVM CODEGEN: {_ENV_COMPILER}={exe!r} is not an executable file.")
    return found


def compile_late_mlir(mlir: str) -> bytes:
    """Compile Late HIVM/AVE MLIR text into an AIV binary.

    Returns the raw bytes of the compiler's output object. Raises CompilerError
    on any failure; there is no fallback path.
    """
    if not isinstance(mlir, str) or not mlir.strip():
        raise CompilerError("HIVM CODEGEN: refusing to compile empty MLIR text")
    _kernel_cache_disabled()
    exe = _resolve_compiler()
    target = os.environ.get(_ENV_TARGET, "").strip() or _DEFAULT_TARGET

    with tempfile.TemporaryDirectory(prefix="tilelang-hivm-") as tmp:
        src = Path(tmp) / "kernel.mlir"
        out = Path(tmp) / "kernel.o"
        src.write_text(mlir, encoding="utf-8")
        cmd = [
            exe,
            str(src),
            f"--target={target}",
            "--input-ir-stage=late-hivm",
            "-o",
            str(out),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            raise CompilerError(f"HIVM CODEGEN: the Late pipeline exceeded {_TIMEOUT_S}s") from exc
        if proc.returncode != 0:
            raise CompilerError(
                "HIVM CODEGEN: the Late pipeline failed "
                f"(exit {proc.returncode}).\n"
                f"command: {' '.join(cmd)}\n"
                f"stderr:\n{proc.stderr[-4000:]}"
            )
        if not out.is_file() or out.stat().st_size == 0:
            raise CompilerError(
                f"HIVM CODEGEN: the Late pipeline exited 0 but wrote no binary.\ncommand: {' '.join(cmd)}\nstderr:\n{proc.stderr[-4000:]}"
            )
        return out.read_bytes()


def _hook(mlir, target) -> bytes:
    """The C++-visible entry. The target is validated C++-side."""
    _ = target
    return compile_late_mlir(str(mlir))


def register(compiler: str | None = None, target: str | None = None) -> None:
    """Register the callback, optionally pinning the compiler and target.

    Registered lazily and explicitly: importing this module has no side effect,
    so a host with no Bisheng still imports tilelang and only the HIVM build
    entry fails, with a reason.
    """
    if compiler is not None:
        os.environ[_ENV_COMPILER] = compiler
    if target is not None:
        os.environ[_ENV_TARGET] = target
    tvm_ffi.register_global_func(_HOOK, _hook, override=True)


def unregister() -> None:
    if tvm_ffi.get_global_func(_HOOK, allow_missing=True) is not None:
        tvm_ffi.remove_global_func(_HOOK)
