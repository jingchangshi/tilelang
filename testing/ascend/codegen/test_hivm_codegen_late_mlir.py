"""P5-2/P5-3: the real HIVM DeviceCodeGen emits Late HIVM/AVE MLIR.

These tests replay the FROZEN Golden input (`golden_codegen_input.json`, the
authoritative P2-replay artefact) through the Source-Only build entry. Nothing
here compiles a binary, loads a device or reads the frozen P3 program, so the
whole file runs on a host with no Bisheng and no NPU.

Two properties are checked, and they are the ones the goal asks for:

  * P5-2 emission: the emitted text is Late HIVM/AVE MLIR for THIS TIR -- the
    Golden's grid, the Golden's UB offsets and the Golden's op subset -- not a
    copy of the reference `.asc` and not text keyed on the Golden's digest.
  * P5-3 provenance: a module built by this entry must never describe itself as
    the frozen P3 PREBUILT artefact.
"""

from pathlib import Path

import pytest

from tilelang import tvm

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "tests/golden/hivm/vadd_f32_single_tile"
GOLDEN = ROOT / "golden_codegen_input.json"

EMIT_ENTRY = "target.build.tilelang_hivm_source_only"
BUILD_ENTRY = "target.build.tilelang_hivm"


def _golden():
    """Load the frozen Golden input and its kernel."""
    raw = GOLDEN.read_text(encoding="utf-8")
    mod = tvm.ir.load_json(raw)
    keys = list(mod.functions)
    assert len(keys) == 1, f"the frozen Golden carries ONE kernel, got {len(keys)}"
    return mod, mod[keys[0]]


def _hivm_target(kernel):
    """The Golden was frozen from a plain `ascend` target.

    The HIVM entries require the explicit `hivm` key, which is the routing guard
    that keeps a plain Ascend/PTO target out of this codegen. The key is added
    here rather than removed from the guard.
    """
    cfg = dict(kernel.attrs["target"].export())
    cfg["keys"] = ["ascend", "hivm"]
    return tvm.target.Target(cfg)


def _emit():
    mod, kernel = _golden()
    emit = tvm.ffi.get_global_func(EMIT_ENTRY)
    return emit(mod, _hivm_target(kernel))


def test_emit_entry_is_registered():
    assert tvm.ffi.get_global_func(EMIT_ENTRY) is not None
    assert tvm.ffi.get_global_func(BUILD_ENTRY) is not None


def test_emits_one_module_for_the_vector_core():
    mlir = _emit()
    assert mlir.count("module attributes") == 1
    # AIV, not a generic "vector core": the Late pipeline's module-level
    # dispatcher accepts only the AIC/AIV/MIX enum and refuses anything else.
    assert "hivm.module_core_type<AIV>" in mlir
    assert mlir.count("func.func @main_kernel(") == 1
    assert "hacc.entry" in mlir
    assert "hacc.function_kind = #hacc.function_kind<DEVICE>" in mlir
    assert "hivm.func_core_type = #hivm.func_core_type<AIV>" in mlir
    # The target must be declared, or the store's l2_cache_mode is rejected and
    # an incomplete device spec crashes a downstream pass.
    assert 'hacc.target = #hacc.target<"Ascend950PR_9579">' in mlir
    assert "dlti.target_system_spec" in mlir


def test_grid_comes_from_the_tir():
    """grid=64 is READ from thread_extent, so the attribute must match it."""
    mod, kernel = _golden()
    grid = int(kernel.attrs["thread_extent"]["blockIdx.x"])
    assert grid == 64
    assert f"hacc.block_dim = {grid} : i64" in _emit()


def test_dma_stage_is_emitted_from_the_tir_offsets():
    mlir = _emit()
    # Two GM->UB loads and one UB->GM store, at the TIR's three UB offsets.
    assert mlir.count("hivm.hir.load ") == 2
    assert mlir.count("hivm.hir.store ") == 1
    for byte_off in (0, 65536, 131072):
        assert f"pointer_cast(%c_ub{byte_off}_)" in mlir, f"missing UB view at {byte_off} B"
    # The store's cache mode is the reviewed one (P1 D012).
    assert "l2_cache_mode = 4 : i32" in mlir


def test_per_block_gm_stride_comes_from_the_tir():
    """The GM offset is blockIdx * 8192, and 8192 = 524288 / 64 is checked."""
    mlir = _emit()
    assert mlir.count("hivm.hir.get_block_idx") == 3
    assert mlir.count("arith.muli %gm_src_0_bid, %c8192_i64 : i64") == 1
    assert mlir.count("arith.muli %gm_dst_2_bid, %c8192_i64 : i64") == 1
    assert "memref<524288xf32, #hivm.address_space<gm>>" in mlir


def test_sync_flags_are_emitted_in_tir_order():
    mlir = _emit()
    mte2 = mlir.index("hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]")
    vf = mlir.index("func.call @main_kernel_simd_vf_0")
    mte3 = mlir.index("hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]")
    assert mte2 < vf < mte3, "the flags must bracket the vector function"


def test_vector_function_is_outlined_and_looped():
    mlir = _emit()
    # MLIR has no nested func.func, so the VF is a sibling of the entry.
    assert mlir.count("func.func @main_kernel_simd_vf_0(") == 1
    assert "hivm.vector_function" in mlir
    assert "ave.hir.pge <ALL>" in mlir
    # The vector body covers the Golden's 128 x 64 lanes as an ELEMENT range:
    # 0 .. 8192 step 64. The Late form indexes elements directly, so the loop
    # bound is the tile and the step is the lane count.
    assert "scf.for %i = %c0 to %c8192_elem step %c64_i" in mlir
    assert mlir.count("ave.hir.vload <NORM>") == 2
    assert mlir.count("ave.hir.vadd ") == 1
    assert mlir.count("ave.hir.masked_store <NORM_B32>") == 1


def test_barrier_closes_the_entry():
    """P1 D013 puts the completion barrier in the ENTRY, not the VF."""
    mlir = _emit()
    assert mlir.count("hivm.hir.pipe_barrier[<PIPE_ALL>]") == 1
    entry_end = mlir.index("hivm.hir.pipe_barrier[<PIPE_ALL>]")
    vf_start = mlir.index("func.func @main_kernel_simd_vf_0(")
    assert entry_end < vf_start, "the barrier belongs to the entry body"


def test_emission_is_deterministic():
    """The same TIR twice must produce identical text, byte for byte."""
    assert _emit() == _emit()


def test_emission_does_not_key_on_the_golden_digest():
    """The emitter must read the TIR, not recognise the Golden.

    The mutation removes the block-index term from every GM address. A semantic
    change like this MUST either change the output or be refused; what it must
    not do is return the unchanged preset text, which is what a generator keyed
    on the Golden's digest would do.
    """
    original = _emit()
    mod, kernel = _golden()
    cfg = dict(kernel.attrs["target"].export())
    cfg["keys"] = ["ascend", "hivm"]
    target = tvm.target.Target(cfg)

    key = list(mod.functions)[0]
    _drop_block_index(mod, key, kernel)
    emit = tvm.ffi.get_global_func(EMIT_ENTRY)
    with pytest.raises(Exception) as excinfo:
        emit(mod, target)
    message = str(excinfo.value).lower()
    assert "unsupported" in message or "hivm codegen" in message

    # The honest input still emits the honest text, unchanged.
    assert _emit() == original


def _drop_block_index(mod, key, kernel):
    """Replace `bx * 8192` with a flat `0` in every GM address.

    The result is a well-formed TIR whose GM accesses no longer depend on the
    block index. The emitter requires that term, so this must be refused.
    """
    from tvm import tirx

    fn = mod[key]

    def rewrite(node):
        if isinstance(node, tirx.Mul):
            sides = [node.a, node.b]
            if any(isinstance(x, tirx.IntImm) and int(x) == 8192 for x in sides):
                return tirx.IntImm("int32", 0)
        return None

    def replace(n):
        hit = rewrite(n)
        return n if hit is None else hit

    body = tirx.stmt_functor.ir_transform(fn.body, None, replace, ["tirx.Mul"])
    mod[key] = tirx.PrimFunc(fn.params, body, fn.ret_type, fn.buffer_map, fn.attrs, fn.span)


def test_generated_format_token_differs_from_prebuilt():
    """P5-3 provenance: the two format tokens must stay distinct."""
    identity = (REPO / "src/ascend/hivm_prebuilt_identity.h").read_text()
    assert 'kHivmPrebuiltFormat = "hivm-aibin"' in identity
    assert 'kHivmGeneratedFormat = "hivm-generated-aibin"' in identity
