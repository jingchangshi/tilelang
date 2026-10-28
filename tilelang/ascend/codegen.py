from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen

# Plain Ascend and PTO share the "ascend" target kind, so they are declared as
# two BackendModules (see backend.py) whose mutually exclusive `supports_target`
# predicates select one; DeviceCodegen itself no longer carries a predicate.
PTO_CODEGEN = DeviceCodegen(
    "pto",
    build=global_func_device_codegen("target.build.tilelang_pto"),
    build_without_compile=global_func_device_codegen("target.build.tilelang_pto_without_compile"),
)

ASCEND_CODEGEN = DeviceCodegen(
    "ascend",
    build=global_func_device_codegen("target.build.tilelang_ascend"),
    build_without_compile=global_func_device_codegen("target.build.tilelang_ascend_without_compile"),
)

# HIVM is the REAL DeviceCodeGen: it turns the prepared TIR into Late HIVM/AVE
# MLIR and compiles that text through the P2 Late pipeline.
#
#   target.build.tilelang_hivm                generate + compile       (GENERATED)
#   target.build.tilelang_hivm_source_only_   generate, return a MODULE  (source only)
#   module
#
# The PREBUILT entry (`target.build.tilelang_hivm_prebuilt`) is deliberately NOT
# reachable from here. It loads the frozen P3 binary and exists only as a
# TEST_ONLY path that a test calls directly, so a normal `hivm` backend can never
# select it -- a build failure raises instead of substituting that binary.
#
# `build_without_compile` needs a MODULE, not text: `DeviceCodegen.lower` returns
# the callback's value into a chain that calls `inspect_source()`. The plain
# `..._source_only` entry returns an `ffi::String`, so it cannot be bound here;
# the `..._source_only_module` entry wraps the same generated MLIR in a source
# module without compiling and without touching the frozen binary.
HIVM_CODEGEN = DeviceCodegen(
    "ascend",
    build=global_func_device_codegen("target.build.tilelang_hivm"),
    build_without_compile=global_func_device_codegen("target.build.tilelang_hivm_source_only_module"),
)
