"""Ascend backend manifests (plain AscendC, PTO and HIVM)."""

from __future__ import annotations

from tilelang.ascend.target import (
    target_is_hivm,
    target_is_plain_ascend,
    target_is_pto,
)
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.module import BackendModule, register_backend
from tilelang.contrib import bisheng
from tilelang.env import TILELANG_TEMPLATE_PATH, env
from tilelang.transform import PassConfigKey

from . import codegen, execution_backend, pipeline


def tilelang_callback_ascend_compile(code, target, pass_config=None):
    """Compile AscendC source into a cached, executable CCE ELF."""
    cfg = pass_config or {}
    extra_flags = bisheng.normalize_options(cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS))
    options = ["-I" + TILELANG_TEMPLATE_PATH, *extra_flags]

    target_arch = bisheng.get_target_npu_arch(target)

    compile_options = bisheng.get_bisheng_compile_options(target_arch, options)
    linker_options = bisheng.get_aibin_linker_options()

    from tilelang.cache.ascend_binary_cache import AscendBinaryCache

    compile_format = AscendBinaryCache.binary_format
    cache_key = AscendBinaryCache.make_key(
        code=code,
        target_kind=target.kind.name,
        target_arch=target_arch,
        compile_format=compile_format,
        options=compile_options,
        linker_options=linker_options,
    )
    cached_binary = AscendBinaryCache.load(cache_key, compile_format)
    if cached_binary is not None:
        return bytearray(cached_binary)

    aibin = bisheng.compile_ascend(
        code,
        target_format=compile_format,
        npu_arch=target_arch,
        options=options,
        verbose=env.get_default_verbose(),
    )
    AscendBinaryCache.save(cache_key, compile_format, aibin)
    return aibin


def _hivm_compile_callback(mlir, target):
    """The registered HIVM compile callback.

    It delegates to `tilelang.contrib.hivm`, which owns the compiler discovery,
    the target consistency check and the failure behaviour. Imported lazily so
    this module has no import-time dependency on the contribution package.

    It enters through `hivm.hook`, the SAME target-aware entry that an explicit
    `hivm.register()` installs. Calling `compile_late_mlir` directly here would
    drop `target` on the floor: the Target is what ties the device this build
    selected to the device the compile command emits for, so bypassing it makes
    the automatic registration weaker than the explicit one for no reason.
    """
    from tilelang.contrib import hivm

    return hivm.hook(mlir, target)


# Plain Ascend and PTO share the "ascend" target kind, so each is its own
# BackendModule with a mutually exclusive `supports_target` predicate; the
# resolver then selects exactly one. Both reuse the same lowering pipeline.
BACKEND = register_backend(
    BackendModule(
        name="ascend",
        target_kinds=("ascend",),
        supports_target=target_is_plain_ascend,
        pipelines={"ascend": pipeline.ascend_pipeline},
        device_codegens={"ascend": codegen.ASCEND_CODEGEN},
        execution_backends=execution_backend.ASCEND_EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
        callbacks={"tilelang_callback_ascend_compile": tilelang_callback_ascend_compile},
    )
)

PTO_BACKEND = register_backend(
    BackendModule(
        name="pto",
        target_kinds=("ascend",),
        supports_target=target_is_pto,
        pipelines={"ascend": pipeline.ascend_pipeline},
        device_codegens={"ascend": codegen.PTO_CODEGEN},
        execution_backends=execution_backend.PTO_EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
    )
)

# The HIVM backend. It shares the "ascend" target kind and is selected by the
# "hivm" key; its pipeline and device-codegen mappings stay keyed by "ascend",
# and it reuses the existing host codegens. It opens tvm_ffi only (goal req 9)
# and runs no AscendC/PTO codegen.
#
# Its device codegen is the GENERATED one (see codegen.HIVM_CODEGEN): the TIR is
# turned into Late HIVM/AVE MLIR and compiled by the P2 Late pipeline. The
# TEST_ONLY PREBUILT entry is deliberately absent from this manifest, so a
# normal `hivm` build can never select it.
#
# cache_reusable=False (BC-1). The original reason was the PREBUILT entry
# injecting a binary read from a fixed PATH. The reason still holds for the
# GENERATED path, for a different cause: the cache key hashes the TIR, target,
# backend, pass configs and flags, but NOT the compiler or the template bitcode
# the device binary is actually derived from. A warm entry would therefore
# describe a binary this build never produced and could not audit. goal P5 §6.3
# keeps the HIVM kernel cache disabled for exactly this phase, so reuse is
# refused rather than trusted. Every other backend keeps the default True.
HIVM_BACKEND = register_backend(
    BackendModule(
        name="hivm",
        target_kinds=("ascend",),
        supports_target=target_is_hivm,
        pipelines={"ascend": pipeline.ascend_pipeline},
        device_codegens={"ascend": codegen.HIVM_CODEGEN},
        execution_backends=execution_backend.HIVM_EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
        cache_reusable=False,
        # Registering the compile callback here is what makes a plain
        # `target="hivm"` build work without the caller having to import and
        # call `tilelang.contrib.hivm.register()` first. The callback resolves
        # the compiler from HIVM_COMPILER at call time, so registering it has no
        # effect on a host with no compiler: only the build entry fails, with a
        # reason naming the variable.
        callbacks={"tilelang_callback_hivm_compile": _hivm_compile_callback},
    )
)
