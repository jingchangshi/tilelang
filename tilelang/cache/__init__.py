"""The cache utils with class and database persistence - Init file"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal
from tvm.target import Target as TVMTarget
from tvm.tirx import PrimFunc
from tilelang.jit import JITKernel
from tilelang import env
from tilelang.jit.adapter.cutedsl.kernel_cache import CuTeDSLKernelCache
from tilelang.jit.adapter.cython.kernel_cache import CythonKernelCache
from tilelang.jit.adapter.nvrtc.kernel_cache import NVRTCKernelCache
from tilelang.jit.adapter.torch.kernel_cache import TorchKernelCache
from tilelang.jit.adapter.kernel_cache import TVMFFIKernelCache
from tilelang.ascend.kernel_cache import AscendCythonKernelCache, AscendTVMFFIKernelCache

if TYPE_CHECKING:
    from .kernel_cache import KernelCache

TargetLike = str | dict[str, object] | TVMTarget

# Create a map of singleton instance of KernelCaches
_dispatch_map: dict[str, KernelCache] = {
    "tvm_ffi": TVMFFIKernelCache(),
    "cython": CythonKernelCache(),
    "nvrtc": NVRTCKernelCache(),
    "cutedsl": CuTeDSLKernelCache(),
    "pto": CythonKernelCache(),
    "torch": TorchKernelCache(),
}

_ascend_dispatch_map: dict[str, KernelCache] = {
    "tvm_ffi": AscendTVMFFIKernelCache(),
    "cython": AscendCythonKernelCache(),
}

# Backend module names that share the Ascend target KIND and must therefore use
# the Ascend cache layout. This is an explicit list, not a `kind == "ascend"`
# test, because PTO deliberately uses the GENERIC map and has no "pto" key in
# _ascend_dispatch_map -- switching every Ascend-kind target over would make PTO
# unresolvable. The literal names match the BackendModule `name=` arguments in
# tilelang/ascend/backend.py.
_ASCEND_CACHE_MODULES = frozenset({"ascend", "hivm"})


def _resolve_cache_dispatch(
    target: TargetLike | None,
    target_host: TargetLike | None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl", "pto"] | None,
    verbose: bool | None,
):
    if target is None:
        target = env.get_default_target()
    if execution_backend is None:
        execution_backend = env.get_default_execution_backend()
    if verbose is None:
        verbose = env.get_default_verbose()

    from tilelang.backend.module import create_backend_context

    requested_backend = execution_backend
    context = create_backend_context(target, target_host, requested_backend)
    resolved_backend = context.execution_backend.name
    if verbose:
        allowed_now = context.module.allowed_execution_backends(context.target, include_unavailable=False)
        if requested_backend in (None, "auto") or requested_backend != resolved_backend:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            logger.info(
                "Execution backend resolved -> '%s' (requested='%s', target='%s', allowed: %s)",
                resolved_backend,
                requested_backend,
                context.target.kind.name,
                ", ".join(sorted(allowed_now)),
            )
    # Dispatch on the resolved backend module rather than on the target kind.
    # The HIVM backend is named "hivm", so a bare `name == "ascend"` test sent
    # it to the GENERIC map and gave an Ascend kernel CUDA `.cu` sidecar names
    # (BC-5). The explicit set covers ascend and hivm only.
    dispatch_map = (_ascend_dispatch_map if context.module.name in _ASCEND_CACHE_MODULES else _dispatch_map)
    if resolved_backend not in dispatch_map:
        raise ValueError(f'Cannot find support for execution backend "{resolved_backend}"')
    return dispatch_map[resolved_backend], context, verbose


def cached(
    func: PrimFunc = None,
    out_idx: list[int] = None,
    *args,
    target: TargetLike | None = None,
    target_host: TargetLike | None = None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl", "pto"] | None = None,
    verbose: bool | None = None,
    pass_configs: dict | None = None,
    compile_flags: list[str] | str | None = None,
) -> JITKernel:
    """
    Caches and reuses compiled kernels (using KernelCache class).
    """
    cache, backend_context, verbose = _resolve_cache_dispatch(target, target_host, execution_backend, verbose)
    return cache.cached(
        func,
        out_idx,
        *args,
        backend_context=backend_context,
        verbose=verbose,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )
