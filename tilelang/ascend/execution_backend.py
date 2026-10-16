from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec

# Plain Ascend prefers tvm_ffi -- it is listed first, so execution_backend="auto"
# resolves to it -- and also accepts cython.
ASCEND_EXECUTION_BACKENDS = [
    ExecutionBackendSpec(
        "tvm_ffi",
        enable_host_codegen=True,
        enable_device_compile=True,
        # Ascend's host codegen lowers the TVM-FFI callee-allocated-output
        # result slot, so a kernel with out_idx allocates its outputs inside
        # the generated host function. This is the single gate for that ABI
        # (see tilelang/jit/abi.py): MakePackedAPI and the adapter both read
        # the attribute it stamps, so the backend declares the capability here
        # instead of each side sniffing the target.
        supports_callee_allocated_outputs=True,
    ),
    ExecutionBackendSpec("cython"),
]

# PTO's execution backend owns the AOT pipeline (ptodsl -> ptoas -> bisheng).
PTO_EXECUTION_BACKENDS = [
    ExecutionBackendSpec("pto"),
]

# HIVM opens tvm_ffi ONLY (goal section 4 req 9). The PREBUILT module reuses the
# existing TVM-FFI host ABI -- the P3 device ABI stays three GM pointers and the
# host side allocates callee-allocated outputs, exactly as plain Ascend does --
# so the same capability flag applies. cython and pto are rejected by their
# absence here.
HIVM_EXECUTION_BACKENDS = [
    ExecutionBackendSpec(
        "tvm_ffi",
        enable_host_codegen=True,
        enable_device_compile=True,
        supports_callee_allocated_outputs=True,
    ),
]
