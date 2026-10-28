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

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import tvm.ffi as tvm_ffi

__all__ = [
    "register",
    "unregister",
    "hook",
    "compile_late_mlir",
    "compiler_identity",
    "CompilerError",
]

_HOOK = "tilelang_callback_hivm_compile"
_ENV_COMPILER = "HIVM_COMPILER"
_ENV_TARGET = "HIVM_TARGET"
_DEFAULT_TARGET = "Ascend950PR_9579"
_TIMEOUT_S = 1800
_PROBE_TIMEOUT_S = 120
_LATE_STAGE_FLAG = "--input-ir-stage"
_LATE_STAGE_VALUE = "late-hivm"


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
    _check_late_capability(found)
    return found


def _check_late_capability(exe: str) -> None:
    """Refuse a compiler that does not declare the `late-hivm` input stage.

    This is a CAPABILITY check, not an identity check: it establishes that the
    executable can run the stage this generator emits for, and it does NOT claim
    the binary is any particular approved build. The compiler's measured
    identity is reported by `compiler_identity()` and recorded by the caller, so
    a wrong-generation compiler is visible without being silently trusted.

    A compiler that cannot be probed is refused rather than assumed capable,
    because "we could not tell" has the same consequence here as "it cannot".
    """
    try:
        proc = subprocess.run(
            [exe, "--help"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CompilerError(
            f"HIVM CODEGEN: {_ENV_COMPILER}={exe!r} could not be probed for the "
            f"`--input-ir-stage=late-hivm` capability ({exc}), so it is refused."
        ) from exc
    help_text = (proc.stdout or "") + (proc.stderr or "")
    if _LATE_STAGE_FLAG not in help_text:
        raise CompilerError(
            f"HIVM CODEGEN: {_ENV_COMPILER}={exe!r} does not declare "
            f"`{_LATE_STAGE_FLAG}` in its options, so it is not a P2 Late "
            "compiler and cannot consume the generated Late MLIR. This callback "
            "never falls back to the frozen PREBUILT binary."
        )
    # Declaring the flag is necessary but not sufficient: the probe has to have
    # SUCCEEDED and the declaration has to include the stage value this emitter
    # writes. A `--help` that exits non-zero, or a compiler that knows the flag
    # with some other enum, would otherwise pass the check above and fail later
    # (or, worse, silently compile under a different stage).
    if proc.returncode != 0:
        raise CompilerError(
            f"HIVM CODEGEN: {_ENV_COMPILER}={exe!r} exited {proc.returncode} "
            f"when probed with `--help`, so its `{_LATE_STAGE_FLAG}` "
            "declaration cannot be trusted and it is refused."
        )
    if _LATE_STAGE_VALUE not in help_text:
        raise CompilerError(
            f"HIVM CODEGEN: {_ENV_COMPILER}={exe!r} declares "
            f"`{_LATE_STAGE_FLAG}` but not the `{_LATE_STAGE_VALUE}` stage "
            "this generator emits for, so it cannot consume the generated Late "
            "MLIR. This callback never falls back to the frozen PREBUILT binary."
        )


def compiler_identity(exe: str | None = None) -> dict[str, object]:
    """Return the MEASURED identity of the configured compiler.

    The digest is computed from the file's bytes at call time. It is recorded
    alongside a compile so a later reader can tell which build produced an
    artefact; it is NOT compared against an allow-list here, because pinning a
    digest would refuse every legitimate rebuild (decision: capability check
    plus recorded identity).
    """
    resolved = Path(exe or _resolve_compiler()).resolve()
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest}


def _resolve_target() -> str:
    """The ONE device target this compile is for, with environment conflicts refused.

    The target is not a free parameter. `_DEFAULT_TARGET` is the single device
    this generator's MLIR declares (`hacc.target`), and the emitted module, the
    compiler's `--target` and the output ELF must all be that device. `HIVM_TARGET`
    therefore acts as an ASSERTION, not as an override:

      * unset              -> the verified default is used
      * equal to default   -> accepted
      * anything else      -> refused before the compiler starts

    Refusing here is the point: letting an environment value silently retarget a
    build would produce an ELF for a device whose legality the MLIR and the
    tests never established.
    """
    requested = os.environ.get(_ENV_TARGET, "").strip()
    if requested and requested != _DEFAULT_TARGET:
        raise CompilerError(
            f"HIVM CODEGEN: {_ENV_TARGET}={requested!r} contradicts the target "
            f"this codegen emits for ({_DEFAULT_TARGET!r}). The module's "
            "`hacc.target`, the compiler's `--target` and the output ELF must "
            "name ONE device, so a differing value is refused before the "
            "compiler starts rather than silently retargeting the build."
        )
    return _DEFAULT_TARGET


def _target_from_ffi(target: object) -> str | None:
    """The ONE device target carried by the TileLang Target handed to the hook.

    Returns None when the value is not a Target (the text entry can be called
    without one). The HIVM backend's target carries no `arch`/`mcpu` attribute,
    so the device is the generator's fixed one; what this function establishes
    is that a target which DOES name a device must name the same one.

    When BOTH `arch` and `mcpu` are present they must AGREE. Returning only the
    first one would let `arch=A, mcpu=B` slip through as if it named A: the
    caller compares one value against the compile device and the disagreement in
    the other attribute is never examined. A target that names two devices has
    not named one, so it is refused here rather than silently reduced.
    """
    attrs = getattr(target, "attrs", None)
    if attrs is None:
        return None
    if not hasattr(attrs, "get"):
        return None
    named = {
        key: str(attrs.get(key))
        for key in ("arch", "mcpu")
        if attrs.get(key) is not None
    }
    if len(set(named.values())) > 1:
        raise CompilerError(
            f"HIVM CODEGEN: the target names two different devices "
            f"({', '.join(f'{k}={v!r}' for k, v in named.items())}). The MLIR, "
            "the compiler argument and the output ELF must name ONE device, so "
            "a target that disagrees with itself is refused."
        )
    return next(iter(named.values())) if named else None


def compile_late_mlir(mlir: str) -> bytes:
    """Compile Late HIVM/AVE MLIR text into an AIV binary.

    Returns the raw bytes of the compiler's output object. Raises CompilerError
    on any failure; there is no fallback path.
    """
    if not isinstance(mlir, str) or not mlir.strip():
        raise CompilerError("HIVM CODEGEN: refusing to compile empty MLIR text")
    _kernel_cache_disabled()
    exe = _resolve_compiler()
    target = _resolve_target()

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
        blob = out.read_bytes()
        # A non-empty output file is NOT evidence that the compiler honoured
        # `--target`. Checking only the command line would assert what we asked
        # for, not what we got, so the artifact itself is inspected: it must be
        # the Ascend AIV object this device produces. Device identity in the
        # artifact is what makes "target / --target / ELF name ONE device" a
        # measurement instead of a restatement of the argv.
        _check_aiv_elf(blob, target, cmd)
        return blob


# ELF shape of an Ascend AIV device program, MEASURED on the frozen P2/P3
# artifacts (all six kernel.o copies: 4448 B, e_ident class 2 / data 1, e_type 2,
# e_machine 0x1029, e_flags 0x990000).
#
# e_type is 2 (ET_EXEC), NOT 1: these device programs carry the extension ".o"
# but are NOT relocatable objects, so a "looks like a .o therefore ET_REL" check
# would reject the real approved artifact. The value below is what the bytes say.
#
# WHAT THIS CHECK DOES AND DOES NOT PROVE. It screens the artifact for Ascend/AIV
# SHAPE. It does NOT decode a device model: nothing establishes that 0x1029 +
# 0x990000 is exclusive to Ascend950PR_9579, so this must not be reported as
# "the ELF is for 9579". The device chain is carried by the input MLIR's
# hacc.target, the compiler argv, and the identity of the compiler that ran --
# not by this header.
_ELF_TYPE_EXEC = 2
_ELF_MACHINE_ASCEND = 0x1029
_ELF_FLAGS_AIV = 0x990000


def _check_aiv_elf(blob: bytes, target: str, cmd: list[str]) -> None:
    """Refuse an artifact that is not an Ascend AIV device program.

    This is the half of the target-consistency claim that the command line cannot
    support: the argv says what we ASKED for, the artifact says what the compiler
    actually produced. A copy of some unrelated file, a text diagnostic written
    to `-o`, or an object for another architecture all pass a "file is
    non-empty" check and are refused here instead.

    It is a SHAPE screen, not a device identification. It settles "is this an
    Ascend AIV device program", not "is this an Ascend950PR_9579 program": no
    source in this environment maps 0x1029/0x990000 to a specific device model.
    The device identity comes from the input MLIR's `hacc.target`, the file the
    compiler was given, and the compiler's own recorded identity -- this header
    is corroboration, never the proof.
    """
    if len(blob) < 64 or blob[:4] != b"\x7fELF":
        raise CompilerError(
            "HIVM CODEGEN: the compiler exited 0 but the output is not an ELF "
            f"file (first bytes {blob[:8]!r}), so it cannot be the AIV binary "
            f"for target {target!r}.\ncommand: {' '.join(cmd)}"
        )
    if blob[4] != 2:
        raise CompilerError(
            f"HIVM CODEGEN: the output ELF is not ELF64 (e_ident[EI_CLASS]={blob[4]}, "
            f"expected 2) for target {target!r}.\ncommand: {' '.join(cmd)}"
        )
    if blob[5] != 1:
        raise CompilerError(
            f"HIVM CODEGEN: the output ELF is not little-endian "
            f"(e_ident[EI_DATA]={blob[5]}, expected 1) for target {target!r}."
        )
    e_type = int.from_bytes(blob[16:18], "little")
    if e_type != _ELF_TYPE_EXEC:
        raise CompilerError(
            f"HIVM CODEGEN: the output ELF has e_type={e_type}, expected "
            f"{_ELF_TYPE_EXEC} (ET_EXEC; the AIV device programs this pipeline "
            f"emits carry \".o\" but are not relocatable objects) for target "
            f"{target!r}.\ncommand: {' '.join(cmd)}"
        )
    e_machine = int.from_bytes(blob[18:20], "little")
    if e_machine != _ELF_MACHINE_ASCEND:
        raise CompilerError(
            f"HIVM CODEGEN: the output ELF has e_machine=0x{e_machine:04x}, "
            f"expected 0x{_ELF_MACHINE_ASCEND:04x} (the Ascend AIV shape this "
            f"pipeline emits). The compile asked for {target!r} but the artifact "
            "is not an Ascend device program at all.\ncommand: {' '.join(cmd)}"
        )
    e_flags = int.from_bytes(blob[48:52], "little")
    if e_flags != _ELF_FLAGS_AIV:
        raise CompilerError(
            f"HIVM CODEGEN: the output ELF has e_flags=0x{e_flags:08x}, expected "
            f"0x{_ELF_FLAGS_AIV:08x} (Ascend AIV) for target {target!r}."
        )


def hook(mlir, target) -> bytes:
    """The C++-visible entry.

    `target` is NOT ignored. It is the TileLang Target the C++ caller resolved,
    so the device it names is compared against the device the compile command
    uses: a hook that compiled for a different device than the target selected
    would produce an ELF for a device this build never established. The C++ side
    has already refused a non-HIVM target; this refuses a device disagreement.
    """
    from_target = _target_from_ffi(target)
    resolved = _resolve_target()
    if from_target is not None and from_target != resolved:
        raise CompilerError(
            f"HIVM CODEGEN: the target names device {from_target!r} but the "
            f"compile command would emit for {resolved!r}. The MLIR, the "
            "compiler argument and the output ELF must name ONE device."
        )
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
    tvm_ffi.register_global_func(_HOOK, hook, override=True)


def unregister() -> None:
    if tvm_ffi.get_global_func(_HOOK, allow_missing=True) is not None:
        tvm_ffi.remove_global_func(_HOOK)
