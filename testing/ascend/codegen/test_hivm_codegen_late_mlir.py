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
    """The GM offset is blockIdx * 8192, and 8192 = 524288 / 64 is checked.

    The block index is minted ONCE and every subview reuses it: the approved
    Late input verifier refuses a module that derives its three subviews from
    three distinct `get_block_idx` values. So one op, three multiplies.
    """
    mlir = _emit()
    assert mlir.count("arith.muli %bx, %c8192_i64 : i64") == 3
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


# ---------------------------------------------------------------------------
# Semantic mutation regressions: an input change MUST change the output or be
# refused. Each case below is VALID TIR -- it survives save_json/load_json and
# structural equality -- that changes what the program computes, so returning
# the unchanged frozen text for it is a false acceptance, not a conservative one.
#
# These are CODE GEN checks. A mutation that failed to CONSTRUCT the TIR would
# prove nothing about the emitter, so `_mutated` asserts the round trip first:
# a parser refusal must never be counted as a codegen refusal.
#
# `_mutated` loads the Golden ONCE and mutates THAT function's body, keeping its
# own `params`. Rewriting the body of one loaded module onto the params of
# another makes every parameter reference in the body a free Var of a different
# module; a name-based GM check would not notice, and JSON round-tripping does
# not detect it either, because each module is internally consistent. Only
# deriving body and params from the same object is faithful.
# ---------------------------------------------------------------------------


def _mutated(transform, node_types):
    """Mutate the Golden's ONE kernel in place and return (module, target).

    Single load, single source of `params` and `body`. The result is proven to
    be reloadable TIR before anything asks the emitter about it.
    """
    from tvm import tirx

    mod, kernel = _golden()
    key = list(mod.functions)[0]
    fn = mod[key]
    body = tirx.stmt_functor.ir_transform(fn.body, None, transform, node_types)
    # params, buffer_map and attrs come from the SAME function the body did.
    mutated = tvm.ir.IRModule(
        {
            tvm.ir.GlobalVar(key.name_hint): tirx.PrimFunc(
                fn.params, body, fn.ret_type, fn.buffer_map, fn.attrs, fn.span
            )
        }
    )
    reloaded = tvm.ir.load_json(tvm.ir.save_json(mutated))
    tvm.ir.assert_structural_equal(reloaded, mutated)
    return reloaded, _hivm_target(kernel)


def _emit_module(mod, target):
    return tvm.ffi.get_global_func(EMIT_ENTRY)(mod, target)


def _op_name(expr):
    """The registered name of a Call's callee, or "" when it is not an Op.

    `str(call.op)` is the Op's REPR ("ir.Op(span=None, ... name=\"tl.simd.vld\"...)"),
    not its name, so comparing it against an op name never matches. Every
    mutation helper goes through this function for that reason.
    """
    op = getattr(expr, "op", None)
    return getattr(op, "name", "") or ""


def test_vf_constant_address_is_refused_not_reemitted():
    """A constant vld address is REFUSED, not silently re-indexed.

    The emitted subview is indexed by the loop variable ALONE (`%a[%i] [64]`),
    so a TIR reading `A[0:64]` on every iteration would otherwise emit text
    identical to the Golden's, which reads `A[i*64]`. Those are different
    programs: at i=1 the mutation reads A[0:64] and the Golden reads A[64:128].

    The near-neighbour matters as much as the refusal: the DMA addresses in this
    same module ARE constants (`[0]`, `[16384]`, `[32768]`) and must stay legal,
    so the refusal is scoped to VF accesses. The untouched Golden still emitting
    is asserted below and by every other test in this file.
    """
    from tvm import tirx

    original = _emit()
    hits = []

    def to_constant(node):
        # The vld is a Bind VALUE, so it is reached here by rebuilding the Bind
        # around a rewritten call rather than by expecting expression recursion.
        if isinstance(node, tirx.Bind):
            value = node.value
            if isinstance(value, tirx.Call) and _op_name(value) == "tl.simd.vld":
                load = value.args[0].args[0]
                if isinstance(load.indices[0], tirx.Mul):
                    hits.append(str(load.indices[0]))
                    flat = tirx.BufferLoad(load.buffer, [tirx.IntImm("int32", 0)])
                    return tirx.Bind(
                        node.var,
                        tirx.Call(
                            value.dtype,
                            value.op,
                            [tirx.address_of(flat), value.args[1]],
                        ),
                    )
        return None

    mod, target = _mutated(to_constant, ["tirx.Bind"])
    # The mutation has to have LANDED, or a passing refusal would be meaningless.
    assert hits, "the vld mutation never applied: the test would prove nothing"

    with pytest.raises(Exception) as excinfo:
        _emit_module(mod, target)
    message = str(excinfo.value)
    assert "constant address" in message, message
    assert "vld" in message

    assert _emit() == original


def test_vadd_mask_operand_is_taken_from_the_call():
    """The vadd predicate comes from the CALL, not from a fixed `%mask`.

    Emitting `%mask` unconditionally describes "add under the ALL predicate" no
    matter which predicate the TIR passed. A predicate this emitter assigned no
    value to -- here an all-false broadcast, which under MODE_ZEROING must zero
    the result -- is an operand the subset cannot honour, so it is refused
    rather than silently replaced by the frozen predicate.
    """
    from tvm import tirx

    original = _emit()
    hits = []

    def to_false_mask(node):
        if isinstance(node, tirx.Bind):
            value = node.value
            if isinstance(value, tirx.Call) and _op_name(value) == "tl.simd.vadd":
                hits.append(str(node.var))
                false_mask = tirx.Broadcast(tirx.IntImm("bool", 0), 256)
                return tirx.Bind(
                    node.var,
                    tirx.Call(
                        value.dtype,
                        value.op,
                        [value.args[0], value.args[1], false_mask, value.args[3]],
                    ),
                )
        return None

    mod, target = _mutated(to_false_mask, ["tirx.Bind"])
    assert hits, "the vadd mask mutation never applied: the test would prove nothing"

    with pytest.raises(Exception) as excinfo:
        _emit_module(mod, target)
    message = str(excinfo.value)
    assert "vadd mask" in message, message
    assert "bound vector value" in message

    assert _emit() == original


def _loop_step_transform(step):
    from tvm import tirx

    def set_step(node):
        if isinstance(node, tirx.For):
            return tirx.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body,
                node.thread_binding,
                node.annotations,
                step=tirx.IntImm("int32", step),
            )
        return None

    return set_step


def test_vf_loop_step_must_be_one():
    """A VF loop with step=2 is refused; the explicit step=1 neighbour is not.

    `scf.for ... step %c64_i` walks ELEMENTS one lane at a time and cannot
    express a TIR step, so a step of 2 would emit the unchanged 128-iteration
    text for a loop the TIR runs 64 times -- a silent change in both iteration
    count and covered range.

    step=1 is the discriminating neighbour: refusing every explicit `step`
    would be a different, wrong fix, and it would reject a module the Golden's
    own semantics allow.
    """
    from tvm import tirx

    original = _emit()

    mod_two, target_two = _mutated(_loop_step_transform(2), ["tirx.For"])
    with pytest.raises(Exception) as excinfo:
        _emit_module(mod_two, target_two)
    message = str(excinfo.value)
    assert "VF loop step 2" in message, message

    # step=1 is semantically "no step" and must still be accepted, unchanged.
    mod_one, target_one = _mutated(_loop_step_transform(1), ["tirx.For"])
    assert _emit_module(mod_one, target_one) == original


def test_kernel_symbol_is_one_name_for_text_and_metadata():
    """The emitted `func.func` name and the runtime FunctionInfo key agree.

    A device PrimFunc with no `global_symbol` used to emit `main_kernel` while
    the runtime metadata was keyed by the GlobalVar name: the build succeeded and
    the launch lookup then failed on a symbol nobody had declared. ONE resolution
    now feeds both, falling back to the GlobalVar name.
    """
    from tvm import tirx
    from tvm.ir import DictAttrs

    mod, kernel = _golden()
    key = list(mod.functions)[0]
    fn = mod[key]
    attrs = {
        name: value
        for name, value in fn.attrs.items()
        if name not in ("global_symbol", "tirx.global_symbol")
    }
    assert "global_symbol" not in attrs, "the mutation must remove the symbol"
    stripped = tirx.PrimFunc(
        fn.params, fn.body, fn.ret_type, fn.buffer_map, DictAttrs(attrs), fn.span
    )
    renamed = tvm.ir.IRModule({tvm.ir.GlobalVar("vadd_renamed"): stripped})
    mod = tvm.ir.load_json(tvm.ir.save_json(renamed))

    mlir = _emit_module(mod, _hivm_target(kernel))
    # The GlobalVar name is the ONE name both sides use -- not a bare "main_kernel".
    assert mlir.count("func.func @vadd_renamed(") == 1
    assert "func.func @main_kernel(" not in mlir
    assert "func.call @vadd_renamed_simd_vf_0(" in mlir


def test_vf_constant_store_address_is_refused_too():
    """The vsts address is guarded like the vld address, not only the load.

    The store is an `Evaluate`, not a `Bind`, and its index is an `Add`
    (`i*64 + 32768`) rather than a bare `Mul`, so a mutation written for the
    load path (Bind + Mul) silently matches NOTHING here and the test would
    pass without ever mutating. That is why this case is separate: it anchors
    on the store's own shape and asserts the mutation LANDED before asserting
    the refusal.
    """
    from tvm import tirx

    original = _emit()
    hits = []

    def to_constant(node):
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call):
            if _op_name(node.value) == "tl.simd.vsts":
                value = node.value
                load = value.args[0].args[0]
                hits.append(str(load.indices[0]))
                flat = tirx.BufferLoad(load.buffer, [tirx.IntImm("int32", 0)])
                return tirx.Evaluate(
                    tirx.Call(
                        value.dtype,
                        value.op,
                        [tirx.address_of(flat), *value.args[1:]],
                    )
                )
        return None

    mod, target = _mutated(to_constant, ["tirx.Evaluate"])
    assert hits, "the vsts mutation never applied: the test would prove nothing"

    with pytest.raises(Exception) as excinfo:
        _emit_module(mod, target)
    message = str(excinfo.value)
    assert "constant address" in message, message
    assert "vsts" in message, message

    assert _emit() == original
