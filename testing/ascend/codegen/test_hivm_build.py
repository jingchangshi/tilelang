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


# Every fake compiler below must ANSWER THE CAPABILITY PROBE, or the probe
# refuses it before the branch under test is ever reached and each test would
# pass for the wrong reason. The probe is `--help`: it must exit 0 and declare
# both `--input-ir-stage` and the `late-hivm` stage value.
_FAKE_PROBE = """if [ "$1" = "--help" ]; then
  echo '  --input-ir-stage=<stage>  Input IR stage: late-hivm, early-hivm'
  exit 0
fi
"""


def _fake_compiler(tmp_path, body: str) -> str:
    fake = tmp_path / "fake-bishengir"
    fake.write_text("#!/bin/sh\n" + _FAKE_PROBE + body, encoding="utf-8")
    fake.chmod(0o755)
    return str(fake)


def test_compiler_failure_is_not_swallowed(tmp_path, monkeypatch):
    """A compiler that exits 0 but writes nothing must still fail the build."""
    monkeypatch.setenv("HIVM_COMPILER", _fake_compiler(tmp_path, "exit 0\n"))
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.compile_late_mlir("module {\n}\n")
    assert "wrote no binary" in str(excinfo.value)


def test_nonzero_exit_is_not_swallowed(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "HIVM_COMPILER", _fake_compiler(tmp_path, "echo 'boom' >&2\nexit 3\n")
    )
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.compile_late_mlir("module {\n}\n")
    message = str(excinfo.value)
    assert "exit 3" in message
    assert "boom" in message


def test_capability_probe_requires_a_successful_help(tmp_path, monkeypatch):
    """`--help` that fails cannot certify the stage, so the compiler is refused."""
    fake = tmp_path / "fake-bishengir"
    fake.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--help\" ]; then\n"
        "  echo '  --input-ir-stage=<stage>  late-hivm'\n"
        "  exit 7\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("HIVM_COMPILER", str(fake))
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.compile_late_mlir("module {\n}\n")
    message = str(excinfo.value)
    assert "exited 7" in message, message


def test_capability_probe_requires_the_late_hivm_stage(tmp_path, monkeypatch):
    """Declaring the FLAG without the `late-hivm` stage is not enough."""
    fake = tmp_path / "fake-bishengir"
    fake.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--help\" ]; then\n"
        "  echo '  --input-ir-stage=<stage>  only early-hivm is supported'\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("HIVM_COMPILER", str(fake))
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.compile_late_mlir("module {\n}\n")
    assert "late-hivm" in str(excinfo.value)


def test_non_elf_output_is_refused(tmp_path, monkeypatch):
    """A diagnostic written to `-o` is not a binary, whatever its exit code."""
    body = 'printf "not an elf at all" > "$5"\nexit 0\n'
    monkeypatch.setenv("HIVM_COMPILER", _fake_compiler(tmp_path, body))
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.compile_late_mlir("module {\n}\n")
    assert "not an ELF" in str(excinfo.value)


def _emit_elf(tmp_path, monkeypatch, *, e_type=2, e_machine=0x1029,
              e_flags=0x990000, pad=256) -> None:
    """Install a fake compiler whose output is an ELF64-LE with these fields.

    The defaults are the MEASURED P2/P3 identity, so the honest case can be
    shown to be accepted and each test then moves exactly one field.
    """
    header = bytearray(pad)
    header[0:4] = b"\x7fELF"
    header[4] = 2  # ELFCLASS64
    header[5] = 1  # little endian
    header[16:18] = e_type.to_bytes(2, "little")
    header[18:20] = e_machine.to_bytes(2, "little")
    header[48:52] = e_flags.to_bytes(4, "little")
    artifact = tmp_path / "fake.elf"
    artifact.write_bytes(bytes(header))
    # argv: $1=src, $2=--target=..., $3=--input-ir-stage=..., $4=-o, $5=output
    body = f'cat "{artifact}" > "$5"\nexit 0\n'
    monkeypatch.setenv("HIVM_COMPILER", _fake_compiler(tmp_path, body))


def test_wrong_machine_artifact_is_refused(tmp_path, monkeypatch):
    """The ELF must be the Ascend AIV SHAPE; another machine is refused.

    This is a shape screen, not a device identification: it settles "is this an
    Ascend device program at all", and the device itself is established by the
    MLIR's `hacc.target` plus the compiler that ran.
    """
    _emit_elf(tmp_path, monkeypatch, e_machine=0x3E)  # x86-64
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.compile_late_mlir("module {\n}\n")
    assert "e_machine=0x003e" in str(excinfo.value)


def test_relocatable_output_is_refused(tmp_path, monkeypatch):
    """These device programs are ET_EXEC even though they are named ".o".

    The measured P2/P3 artifacts are e_type=2. A compiler emitting ET_REL has
    produced something this pipeline never accepted, so it is refused -- and the
    check must not be "it ends in .o therefore relocatable".
    """
    _emit_elf(tmp_path, monkeypatch, e_type=1)
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.compile_late_mlir("module {\n}\n")
    assert "e_type=1" in str(excinfo.value)


def test_wrong_flags_artifact_is_refused(tmp_path, monkeypatch):
    _emit_elf(tmp_path, monkeypatch, e_flags=0x1234)
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.compile_late_mlir("module {\n}\n")
    assert "e_flags=0x00001234" in str(excinfo.value)


def test_measured_aiv_identity_is_accepted(tmp_path, monkeypatch):
    """The control for the three refusals above.

    Without this, "everything is refused" would also make them pass. The
    defaults here ARE the frozen artifacts' measured header, so this asserts the
    checker accepts the artifact this pipeline actually produces.
    """
    _emit_elf(tmp_path, monkeypatch)
    blob = hivm.compile_late_mlir("module {\n}\n")
    assert blob[:4] == b"\x7fELF"
    assert len(blob) == 256


def test_kernel_cache_must_be_off(monkeypatch, tmp_path):
    """Goal §6.3: a cache hit would hide a fresh compile, so it is refused."""
    monkeypatch.delenv("TILELANG_CLEAR_CACHE", raising=False)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.compile_late_mlir("module {\n}\n")
    assert "TILELANG_CACHE_DIR" in str(excinfo.value)


# ---------------------------------------------------------------------------
# R1/R2 wiring: the FORMAL backend path.
#
# The tests above exercise the entries and the callback directly. These assert
# the wiring that a real user build goes through: the backend manifest that the
# resolver selects, the conflict guard that keeps a pto+hivm target out, and the
# source-only MODULE entry that `build_without_compile` returns.
# ---------------------------------------------------------------------------


def test_hivm_backend_is_selected_by_the_hivm_key():
    """A plain `ascend` target resolves to AscendC/PTO; the `hivm` key to HIVM.

    All three share the target kind, so this is the routing guard that decides
    which device codegen runs. It is asserted through the SAME resolver a real
    build uses, not by reading the manifest by name.
    """
    from tilelang.backend.module import create_backend_context

    def keys_for(extra):
        cfg = {"kind": "ascend", "tag": "", "keys": ["ascend", *extra]}
        return create_backend_context(cfg).module.name

    assert keys_for([]) == "ascend"
    assert keys_for(["pto"]) == "pto"
    assert keys_for(["hivm"]) == "hivm"


def test_conflicting_pto_and_hivm_selectors_are_refused():
    """A target carrying BOTH selectors is refused, not silently resolved.

    They name two different device backends. The conflict guard exists because
    each predicate declines the other's key, so the generic multiple-matches
    error cannot be relied on to catch this: resolution could otherwise succeed
    by falling through to a single candidate.
    """
    from tilelang.backend.module import create_backend_context

    cfg = {"kind": "ascend", "tag": "", "keys": ["ascend", "pto", "hivm"]}
    with pytest.raises(ValueError) as excinfo:
        create_backend_context(cfg)
    message = str(excinfo.value)
    assert "Conflicting backend selectors" in message, message
    assert "pto" in message and "hivm" in message


def test_hivm_backend_kernel_cache_is_not_reusable():
    """P5 6.3: a reusable cache entry would hide a fresh compile.

    The cache key does not cover the compiler or the template bitcode the device
    binary is derived from, so reuse would describe a binary this build never
    produced.
    """
    from tilelang.backend.module import get_backend

    assert get_backend("hivm").cache_reusable is False
    # The other backends keep the default; only HIVM is restricted.
    assert get_backend("ascend").cache_reusable is True
    assert get_backend("pto").cache_reusable is True


def test_source_only_module_entry_carries_the_generated_mlir():
    """`build_without_compile` returns a MODULE whose source is the MLIR.

    Both `lower()` paths read `inspect_source()`, so an `ffi::String` entry
    could not serve as this callback. What must hold is that the module carries
    the SAME generated text the text entry returns -- and that it does so with no
    compiler present.
    """
    SOURCE_ONLY = "target.build.tilelang_hivm_source_only"
    SOURCE_ONLY_MODULE = "target.build.tilelang_hivm_source_only_module"

    mod, target = _module_and_target()
    text = tvm.ffi.get_global_func(SOURCE_ONLY)(mod, target)
    source_mod = tvm.ffi.get_global_func(SOURCE_ONLY_MODULE)(mod, target)

    assert hasattr(source_mod, "inspect_source"), (
        "the source-only MODULE entry must return an IRModule, not a str"
    )
    carried = source_mod.inspect_source("ascend")
    assert carried == text, "the module must carry the same MLIR the text entry returns"
    assert "hivm.module_core_type<AIV>" in carried
    # No compiler and no device are involved: this is the semantics-test path.
    assert "func.func @main_kernel(" in carried


# ---------------------------------------------------------------------------
# R2: target consistency between the MLIR, the compiler argv and the Target.
#
# `hivm.hook` takes the Target the C++ side resolved. Ignoring it would let a
# build compile for a device other than the one it selected, so the device the
# command would use and the device the Target names must be ONE.
# ---------------------------------------------------------------------------


def test_hook_refuses_a_target_whose_two_attributes_disagree():
    """`arch` and `mcpu` naming DIFFERENT devices is refused.

    Taking the first present attribute would compare one value against the
    compile device and never examine the other, so `arch=<supported>,
    mcpu=<other>` would pass as if it named the supported device. A target that
    names two devices has not named one.
    """
    target = tvm.target.Target({
        "kind": "ascend", "tag": "",
        "arch": "Ascend950PR_9579",
        "mcpu": "Ascend310P",
        "keys": ["ascend", "hivm"],
    })
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.hook("module {\n}\n", target)
    message = str(excinfo.value)
    assert "two different devices" in message, message
    assert "Ascend950PR_9579" in message and "Ascend310P" in message


def test_hook_accepts_arch_and_mcpu_naming_the_same_device(tmp_path, monkeypatch):
    """The control: agreeing attributes are not a conflict."""
    _emit_elf(tmp_path, monkeypatch)
    target = tvm.target.Target({
        "kind": "ascend", "tag": "",
        "arch": "Ascend950PR_9579",
        "mcpu": "Ascend950PR_9579",
        "keys": ["ascend", "hivm"],
    })
    blob = hivm.hook("module {\n}\n", target)
    assert blob[:4] == b"\x7fELF"


def test_hook_refuses_a_target_naming_another_device(monkeypatch):
    """A Target whose arch contradicts the compile device is refused."""
    target = tvm.target.Target({"kind": "ascend", "tag": "",
                               "arch": "Ascend310P", "keys": ["ascend", "hivm"]})
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.hook("module {\n}\n", target)
    message = str(excinfo.value)
    assert "Ascend310P" in message, message
    assert "ONE device" in message


def test_hook_accepts_a_target_naming_the_same_device(tmp_path, monkeypatch):
    """The control: the SAME device is not a conflict.

    Without this, 'everything is refused' would also make the test above pass.
    """
    _emit_elf(tmp_path, monkeypatch)
    target = tvm.target.Target({"kind": "ascend", "tag": "",
                               "arch": "Ascend950PR_9579",
                               "keys": ["ascend", "hivm"]})
    blob = hivm.hook("module {\n}\n", target)
    assert blob[:4] == b"\x7fELF"


def test_hivm_target_env_contradiction_is_refused(tmp_path, monkeypatch):
    """`HIVM_TARGET` is an ASSERTION, not an override.

    Letting an environment value silently retarget a build would emit an ELF for
    a device whose legality the MLIR never established. A capable compiler is
    installed first, so this reaches the target check instead of stopping at the
    compiler check -- otherwise it would pass for an unrelated reason.
    """
    _emit_elf(tmp_path, monkeypatch)
    monkeypatch.setenv("HIVM_TARGET", "Ascend310P")
    with pytest.raises(hivm.CompilerError) as excinfo:
        hivm.compile_late_mlir("module {\n}\n")
    message = str(excinfo.value)
    assert "contradicts the target" in message, message


def test_hivm_target_env_agreeing_with_default_is_accepted(tmp_path, monkeypatch):
    """The control for the refusal above."""
    _emit_elf(tmp_path, monkeypatch)
    monkeypatch.setenv("HIVM_TARGET", "Ascend950PR_9579")
    blob = hivm.compile_late_mlir("module {\n}\n")
    assert blob[:4] == b"\x7fELF"


def test_compiler_identity_is_the_measured_bytes(tmp_path, monkeypatch):
    """The identity is MEASURED from the file, not declared.

    A reader needs to tell which compiler produced an artefact; a digest that
    was never computed would be an unverifiable claim.
    """
    fake = _fake_compiler(tmp_path, "exit 0\n")
    identity = hivm.compiler_identity(fake)
    measured = hashlib.sha256(Path(fake).read_bytes()).hexdigest()
    assert identity["sha256"] == measured
    assert identity["bytes"] == Path(fake).stat().st_size
    assert Path(str(identity["path"])).name == "fake-bishengir"


# ---------------------------------------------------------------------------
# R4: what a SOURCE-ONLY module is, and what it is not.
#
# The source path exists so the semantics tests run with no compiler and no
# device. It is NOT a restorable artifact: `CSourceModule` serializes a source
# stream while `AscendModuleLoadFromBytes` reads an AscendModule struct, so the
# device loader does not accept it -- for plain AscendC exactly as for HIVM.
# That is asserted as a measured limit below, not papered over; the module that
# DOES round trip is the COMPILED one, and it is tested separately.
# ---------------------------------------------------------------------------


def test_source_only_module_is_a_source_module_not_a_device_module():
    """The source-only entry returns a SOURCE module, and that is observable.

    `build_without_compile` must not hand back something that can be mistaken
    for the compiled device module: the source path exists precisely so the
    semantics tests run with no compiler and no device.

    Two facts are asserted, and the second one is a MEASURED limitation rather
    than a feature. The module's write format is the source format, so
    `ffi.Module.load_from_bytes.asc` -- the DEVICE-module loader -- does not
    accept it. That is not specific to HIVM: the plain AscendC source module
    (format "asc") fails the same way, because `CSourceModule` serializes a
    source stream and the asc loader reads an AscendModule struct. Restoring a
    source-only module is therefore NOT supported, and asserting a round trip
    here would be asserting behaviour the runtime does not have.
    """
    SOURCE_ONLY_MODULE = "target.build.tilelang_hivm_source_only_module"
    source_mod = tvm.ffi.get_global_func(SOURCE_ONLY_MODULE)(*_module_and_target())

    assert "late-hivm-mlir" in source_mod.get_write_formats(), (
        "the source-only module must identify itself by its SOURCE format"
    )
    assert GENERATED_FMT not in source_mod.get_write_formats(), (
        "a source module must not claim the compiled-binary format"
    )

    # Saving is well defined; loading it through the DEVICE-module loader is not.
    blob = tvm.ffi.get_global_func(SAVE_ENTRY)(source_mod)
    assert len(blob) > 0
    with pytest.raises(Exception):
        tvm.ffi.get_global_func(LOAD_ENTRY)(blob)


def test_compiled_module_round_trips_and_keeps_its_generated_token(built):
    """The COMPILED module is the one that must survive a round trip.

    This is the counterpart to the source-module test above: the device module
    really is an `AscendModule`, so the loader reconstructs it, and the
    GENERATED token must survive rather than being rebuilt into the PREBUILT
    description (BC-4).
    """
    blob = tvm.ffi.get_global_func(SAVE_ENTRY)(built)
    restored = tvm.ffi.get_global_func(LOAD_ENTRY)(blob)

    # WHAT IS OBSERVABLE AFTER A RESTORE, measured rather than assumed:
    #
    #   * `inspect_source()` with no format returns "" -- the restored module's
    #     source map is deliberately empty, and only a PREBUILT token
    #     reconstructs a label (BC-4). So an EMPTY label is the correct result
    #     here; asserting "GENERATED in label" would demand a label this
    #     runtime never rebuilds, and asserting only "not PREBUILT" would pass
    #     even if the token had been lost.
    #   * The evidence that the token SURVIVED is therefore taken from the
    #     RESTORED module's own serialization, not from the original blob.
    #
    # Asserting on `blob` (the ORIGINAL module's bytes) would only re-state that
    # the first save contained the token; it would not show the restore kept it.
    restored_blob = tvm.ffi.get_global_func(SAVE_ENTRY)(restored)
    assert GENERATED_FMT.encode() in restored_blob, (
        "the restored module must still serialize under the GENERATED token"
    )
    assert PREBUILT_FMT.encode() not in restored_blob, (
        "a restored GENERATED module must not become PREBUILT"
    )

    # NOTE: `restored.inspect_source(GENERATED_FMT)` is NOT called here. For the
    # GENERATED token that accessor returns the module's CODE -- raw device
    # bytes, which are not valid UTF-8 -- so reading it raises. The token's
    # survival is established by `restored_blob` above instead, which is the
    # stronger statement anyway: it is the restored module's OWN serialization.

    # The PREBUILT answer must stay empty: only a PREBUILT token reconstructs a
    # label (BC-4), and this module must never acquire one.
    prebuilt_answer = restored.inspect_source(PREBUILT_FMT)
    assert prebuilt_answer in (None, ""), (
        "a GENERATED module must not answer the PREBUILT format with a label"
    )
    assert FROZEN_SHA256 not in (prebuilt_answer or "")
    assert "p2-2026" not in (prebuilt_answer or "")


# ---------------------------------------------------------------------------
# The FORMAL context path: `BackendContext.codegen_device` in BOTH modes.
#
# Resolving the backend only proves which manifest was chosen; what a real build
# calls is `codegen_device(mod, compile_device=...)`, so both arms are driven.
#
# The compile arm asserts the ROUTE: the formal device-gen entry emits the Late
# MLIR and looks up the registered `tilelang_callback_hivm_compile` hook. That
# hook is REPLACED here with a controllable one so the route is observable
# without a Bisheng build -- replacing an FFI global is not "the manifest's own
# callback", and the real compiler is covered separately by the `built` tests.
#
# Every test that touches the process-global hook SAVES and RESTORES it, so the
# suite is order-independent whether or not a previous test left one installed.
# ---------------------------------------------------------------------------


HOOK = "tilelang_callback_hivm_compile"


@pytest.fixture
def hook_guard():
    """Save the process-global compile hook and restore it afterwards.

    The hook is process-global state. Removing it in a `finally` without
    restoring the previous value would make later tests pass or fail depending
    on order, and would silently unregister the manifest's own callback for the
    rest of the session.
    """
    previous = tvm.ffi.get_global_func(HOOK, allow_missing=True)
    try:
        yield
    finally:
        tvm.ffi.remove_global_func(HOOK)
        if previous is not None:
            tvm.ffi.register_global_func(HOOK, previous, override=True)


def _context():
    from tilelang.backend.module import create_backend_context

    mod, _target = _module_and_target()
    kernel = mod[list(mod.functions)[0]]
    cfg = dict(kernel.attrs["target"].export())
    cfg["keys"] = ["ascend", "hivm"]
    return create_backend_context(cfg), mod


def test_codegen_device_without_compilation_returns_the_generated_module(
    monkeypatch, hook_guard
):
    """compile_device=False must succeed with NO compiler configured.

    The compiler variable is REMOVED and the hook unregistered for this test, so
    a suite run under a real `HIVM_COMPILER` still proves the source arm needs
    neither. Asserting that without clearing them would inherit the environment
    and prove nothing about the no-compiler path.
    """
    monkeypatch.delenv("HIVM_COMPILER", raising=False)
    tvm.ffi.remove_global_func(HOOK)
    context, mod = _context()
    out = context.codegen_device(mod, compile_device=False)

    # It must be a MODULE, not the ffi::String the text entry returns. The
    # concrete type is `ffi.Module` (NOT tvm.ir.IRModule -- asserted, not
    # assumed, because getting this wrong is exactly the kind of "requirement"
    # a test can encode by mistake); what makes it usable is inspect_source.
    assert isinstance(out, tvm.ffi.Module), (
        f"codegen_device must return a Module, got {type(out).__name__}"
    )
    assert not isinstance(out, str)
    source = out.inspect_source("ascend")
    assert source, "the source arm must carry the generated MLIR"
    assert "hivm.module_core_type<AIV>" in source
    assert "func.func @main_kernel(" in source


def test_codegen_device_with_compilation_uses_the_registered_callback(hook_guard):
    """compile_device=True reaches the registered hook and yields GENERATED.

    A controllable hook is installed in place of the real compiler, so this
    asserts the ROUTE -- the formal entry emits the MLIR, looks the hook up, and
    the bytes it returns become the module -- without depending on a Bisheng
    build. The separate `built` tests cover the real compiler end to end.
    """
    seen = {}

    def fake_hook(mlir, target):
        seen["mlir"] = str(mlir)
        seen["target"] = target
        return b"\x7fELF" + b"\x00" * 8

    tvm.ffi.register_global_func(HOOK, fake_hook, override=True)
    context, mod = _context()
    out = context.codegen_device(mod, compile_device=True)

    # The callback was actually invoked, and with the emitted Late MLIR.
    assert "hivm.module_core_type<AIV>" in seen.get("mlir", "")
    assert "func.func @main_kernel(" in seen["mlir"]
    # And the RESULT is the generated DEVICE module: the provenance label is the
    # observable that distinguishes it from both a source module and the frozen
    # PREBUILT artefact. (`get_write_formats()` is empty for this module type, so
    # it cannot serve as the evidence.)
    label = out.inspect_source()
    assert "GENERATED" in (label or ""), f"not a GENERATED module: {label!r}"
    assert "PREBUILT" not in (label or "")
    assert FROZEN_SHA256 not in (label or "")


def test_codegen_device_with_compilation_fails_without_a_callback(hook_guard):
    """No callback registered means no binary -- never a PREBUILT substitute."""
    tvm.ffi.remove_global_func(HOOK)
    context, mod = _context()
    with pytest.raises(Exception) as excinfo:
        context.codegen_device(mod, compile_device=True)
    message = str(excinfo.value)
    assert "tilelang_callback_hivm_compile" in message, message


# ---------------------------------------------------------------------------
# Engine entry + the sibling routes that must NOT be affected.
#
# `tilelang.engine.lower.device_codegen_without_compile` is the function a real
# `lower()` calls for the source arm; testing `context.codegen_device` alone
# would leave the engine wrapper (which runs LowerIntrin/Simplify/
# HoistBroadcastValues first) unproven. The AscendC and PTO manifests share the
# target kind with HIVM, so their route resolution is asserted here too -- a
# routing regression would silently send a plain `ascend` build through the
# HIVM device codegen.
# ---------------------------------------------------------------------------


def test_engine_device_codegen_without_compile_reaches_the_hivm_source_module(
    monkeypatch, hook_guard
):
    """The engine wrapper produces the HIVM source module with no compiler."""
    monkeypatch.delenv("HIVM_COMPILER", raising=False)
    tvm.ffi.remove_global_func(HOOK)

    from tilelang.engine.lower import device_codegen_without_compile

    context, mod = _context()
    out = device_codegen_without_compile(mod, context)

    assert isinstance(out, tvm.ffi.Module)
    source = out.inspect_source("ascend")
    assert source, "the engine source arm must carry the generated MLIR"
    assert "hivm.module_core_type<AIV>" in source
    assert "func.func @main_kernel(" in source


def _build_via(entry, keys):
    """Call a REGISTERED build entry on the Golden TIR, needing no NPU.

    The Golden module is the same input the HIVM route uses, so each entry is
    exercised on real TIR rather than merely resolved.
    """
    mod, _target = _module_and_target()
    kernel = mod[list(mod.functions)[0]]
    cfg = dict(kernel.attrs["target"].export())
    cfg["keys"] = keys
    fn = tvm.ffi.get_global_func(entry)
    return fn(mod, tvm.target.Target(cfg))


def test_ascendc_entry_actually_generates_ascendc_source():
    """The plain AscendC entry RUNS and emits AscendC, not Late-HIVM MLIR.

    Executing the entry (not just resolving it) is the routing guard: if the
    manifests were ever confused, an 'ascend' build would emit HIVM MLIR and
    fail far from the cause.
    """
    out = _build_via("target.build.tilelang_ascend_without_compile", ["ascend"])
    source = out.inspect_source() or ""
    assert "#include <tl_templates/ascend/" in source
    assert "hivm.module_core_type" not in source, (
        "the AscendC entry must not emit HIVM MLIR"
    )


def test_pto_entry_actually_generates_pto_source():
    """The PTO entry RUNS and emits PTO DSL, not AscendC and not HIVM."""
    out = _build_via("target.build.tilelang_pto_without_compile", ["ascend", "pto"])
    source = out.inspect_source() or ""
    assert "from ptodsl import pto" in source
    assert "hivm.module_core_type" not in source
    assert "#include <tl_templates/ascend/" not in source


def test_prebuilt_entry_actually_produces_its_own_bytes_and_is_not_hivm():
    """The PREBUILT entry RUNS and is genuinely a different code path.

    Wrapper identity and __name__ are deliberately NOT used as evidence: both
    entries expose a Python function called 'build', so identity of the
    wrapper says nothing about what the entry does. What is asserted is the
    OUTPUT -- the PREBUILT entry emits its own TEST_ONLY module, and the
    generated route on the same input REFUSES without a compiler instead of
    silently substituting those bytes.
    """
    prebuilt = _build_via("target.build.tilelang_hivm_prebuilt", ["ascend", "hivm"])
    label = prebuilt.inspect_source() or ""
    assert "TEST_ONLY PREBUILT" in label
    assert "GENERATED device binary" not in label
    # The PREBUILT entry INJECTS the frozen binary, so its label is expected to
    # name that digest. Asserting its absence was wrong: the honest statement is
    # that the digest appears, together with an explicit denial of generation.
    assert FROZEN_SHA256 in label, (
        "the PREBUILT label must name the frozen digest it injects"
    )
    assert "NOT DeviceCodeGen output" in label

    # The generated route on the same input is a DIFFERENT thing, and the
    # distinction must hold whatever the environment: either it produces the
    # GENERATED binary, or -- with no compiler -- it refuses. What it must
    # never do is hand back these prebuilt bytes.
    generated = tvm.ffi.get_global_func("target.build.tilelang_hivm")
    mod, _target = _module_and_target()
    kernel = mod[list(mod.functions)[0]]
    cfg = dict(kernel.attrs["target"].export())
    cfg["keys"] = ["ascend", "hivm"]
    try:
        got = generated(mod, tvm.target.Target(cfg))
    except Exception as exc:
        # No compiler configured: refusing is the required behaviour.
        assert "HIVM_COMPILER" in str(exc), str(exc)
    else:
        got_label = got.inspect_source() or ""
        assert "GENERATED device binary" in got_label, got_label
        assert "TEST_ONLY PREBUILT" not in got_label, (
            "the generated route must never answer with the PREBUILT module"
        )
