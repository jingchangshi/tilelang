// \file hivm_codegen.cc
// \brief Real DeviceCodeGen for the HIVM/HIVMAVE Late dialect.
//
// WHAT THIS IS
// A thin TIR visitor plus an MLIR text emitter. It reads a PREPARED TileLang
// device module (already through `_prepare_device_codegen_mod`, so no
// preparation pass runs again here) and emits Late HIVM/AVE MLIR text. That
// text is the input of the existing AscendNPU-IR Late pipeline; this file never
// runs that pipeline and never touches a device.
//
// WHAT THIS IS NOT
//   * It is NOT the PREBUILT path. It never reads a frozen binary and has no
//     fallback: every failure is an error that carries a reason.
//   * It is NOT a copy of tilelang-mlir-ascend. The supported set is the frozen
//     Golden's op subset and nothing else.
//   * It does NOT re-decide hardware choices. UB placement, loop structure and
//     synchronisation are already in the TIR; nothing here re-derives them, and
//     no HFusion / PlanMemory / GraphSyncSolver pass is called.
//
// THE ARENA MODEL (read this before changing any address)
// The prepared TIR declares ONE arena
//   buf_dyn_shmem : uint8[196608]            scope shared.dyn
// and three aliases `a`, `b`, `c` that all point at the SAME data var from
// element 0. Which alias is meant is therefore NOT recoverable from the alias
// name, and it is not recovered that way: every access carries its offset
// explicitly (`0`, `16384`, `32768` through the vector ops, or the third
// `tvm_access_ptr` argument for a DMA). The emitter reads those offsets and
// never invents a slot base.
//
// UNITS (three systems, deliberately kept apart)
//   * GM offset ......... ELEMENTS  (bx * 8192), emitted via a GM subview
//   * UB offset ......... ELEMENTS, converted to BYTES by dtype_bytes
//   * DMA burst/extent .. BYTES     (32768)
// An element count is never added to, or compared with, a byte count.
//
// SUPPORTED SUBSET (P5-2: the frozen Golden's shape only)
// clang-format off
//   AttrStmt(thread_extent, blockIdx.x)     -> hacc.entry + block_dim
//   AllocBuffer(uint8, shared.dyn)          -> the dynamic UB arena
//   DeclBuffer(shared.dyn)                  -> an alias of that arena
//   SBlock("VECTOR")                        -> the entry body
//   SBlock("SIMD_VF")                       -> an outlined vector function + its call
//   tl.simd.pset                            -> ave.hir.pge <ALL> : vector<256xi1>
//   For(serial)                             -> scf.for
//   Bind(x = vector value)                  -> the consumed computation
//   tl.simd.vld / vadd / vsts               -> ave.hir.vload / vadd / masked_store
//   ascend_copy_gm_to_ubuf / ubuf_to_gm     -> hivm.hir.load / hivm.hir.store
//   ascend_set_flag / ascend_wait_flag      -> hivm.hir.set_flag / wait_flag
// clang-format on
//
// Anything else is refused with the node type and the enclosing function. A
// node is never skipped in silence. Function-level and sblock-level attributes
// are not read at all, so a new `tl.*` hint cannot silently change what the
// emitted module means.
//

#include "hivm_codegen.h"

#include <tvm/ir/expr.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace codegen {
namespace hivm {

using namespace tirx;
using ffi::GetRef;

// Raised for any input outside the supported subset. The message carries the
// node type and the enclosing function so the diagnostic can be acted on
// without re-reading the TIR by hand (goal P5 section 5.5).
class Unsupported : public std::runtime_error {
public:
  explicit Unsupported(const std::string &msg) : std::runtime_error(msg) {}
};

[[noreturn]] void Fail(const std::string &where, const std::string &what) {
  throw Unsupported("HIVM CodeGen: unsupported " + what + " in " + where);
}

// A PrimExpr is not convertible to std::string, and streaming it is the only
// supported way to show it. Diagnostics quote the expression verbatim so the
// reader can find it in the TIR dump without a second translation.
std::string Show(const PrimExpr &e) {
  std::ostringstream os;
  os << e;
  return os.str();
}

std::string Ty(const DataType &dt) {
  std::ostringstream os;
  os << dt;
  return os.str();
}

// Bytes per element for the dtypes this generator knows. An unknown dtype is a
// refusal, not a silent 1-byte assumption: a wrong factor would move every UB
// address in the module.
int64_t DtypeBytes(const DataType &dt) {
  if (dt.is_float() && dt.bits() == 32 && dt.lanes() == 1) {
    return 4;
  }
  if (dt.is_uint() && dt.bits() == 8 && dt.lanes() == 1) {
    return 1;
  }
  Fail("the type table", "dtype " + Ty(dt));
}

// Fixed by the prepared TIR (256-bit vector = 64 float32 lanes). Changing any
// of these would change the hardware instruction, so none is a parameter.
constexpr int64_t kLanes = 64;
constexpr int64_t kMaskLanes = 256;
constexpr int64_t kTotalElements = 524288;  // A / B / C extent
constexpr int64_t kArenaBytes = 196608;     // dyn_shared_memory_buf
constexpr int64_t kPerBlockElements = 8192; // kTotalElements / grid
constexpr int64_t kDmaBytes = 32768;        // 8192 f32 elements
constexpr int64_t kGrid = 64;

// The one device the supported subset targets. Named once so the module
// attribute and the diagnostics cannot drift apart.
constexpr const char *kLateTargetName = "Ascend950PR_9579";

// SSA names for the two stride constants. They spell the role, not the value:
// the Late validator reads a `%c<N>` style name as the constant it stands for,
// so a name carrying the wrong digits (a lane constant named for the block
// stride) is misread as the stride itself.
inline std::string BlockStrideName() { return "c8192_i64"; }
inline std::string LaneStrideName() { return "c64_i"; }

// UB slot bases, in ELEMENTS, as the frozen Golden writes them. They are
// checked against the TIR offsets, never substituted for them.
constexpr int64_t kSlotA = 0;
constexpr int64_t kSlotB = 16384;
constexpr int64_t kSlotC = 32768;

// The canonical f32 SIMD op names, as registered by the Ascend dialect.
constexpr const char *kOpPset = "tl.simd.pset";
constexpr const char *kOpVld = "tl.simd.vld";
constexpr const char *kOpVadd = "tl.simd.vadd";
constexpr const char *kOpVsts = "tl.simd.vsts";

// A pipe pair is one string in the TIR and two pipes in MLIR. Only the pairs
// the frozen Golden uses are mapped; an unknown pair is a refusal, not a guess.
struct FlagPipes {
  const char *from;
  const char *to;
};
const std::unordered_map<std::string, FlagPipes> kFlagPipes = {
    {"MTE2_V", {"PIPE_MTE2", "PIPE_V"}},
    {"V_MTE3", {"PIPE_V", "PIPE_MTE3"}},
};

// The op name of a call, or an empty string when the callee is not an Op (for
// example a GlobalVar to another TIR function, which this subset never calls).
std::string OpName(const CallNode *call) {
  if (auto op = call->op.as<Op>()) {
    return op.value()->name;
  }
  return std::string();
}

// ---------------------------------------------------------------------------

class Emitter {
public:
  explicit Emitter(const PrimFunc &f) : fn_(f) {
    name_ = "main_kernel";
    if (auto s = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
      name_ = static_cast<std::string>(s.value());
    }
  }

  std::string Run() {
    EmitEntry();
    return out_.str();
  }

private:
  // -- output helpers -------------------------------------------------------

  void Indent() { out_ << std::string(2 * depth_, ' '); }

  int64_t Const(const PrimExpr &e, const std::string &what) const {
    if (auto v = e.as<IntImmNode>()) {
      return v->value;
    }
    Fail(scope_,
         what + " (expected an integer constant, got `" + Show(e) + "`)");
  }

  std::string Str(const PrimExpr &e, const std::string &what) const {
    if (const auto *s = e.as<StringImmNode>()) {
      return s->value;
    }
    Fail(scope_, what + " (expected a string literal, got `" + Show(e) + "`)");
  }

  // -- pointer arguments ----------------------------------------------------

  // After `_prepare_device_codegen_mod` every pointer is
  // `tirx.builtin.address_of(BufferLoad(buf, [index]))`. The buffer's DATA VAR
  // names the arena or the kernel parameter; the INDEX is the only thing that
  // says which region is meant, because all three Golden aliases share one data
  // var. The index is kept as a PrimExpr, never folded to a constant here.
  struct Ptr {
    std::string data;   // data var name
    std::string buffer; // the synthesised buffer name, for diagnostics only
    PrimExpr index;     // ELEMENT index
  };

  Ptr AddressOf(const PrimExpr &arg, const std::string &what) {
    const auto *call = arg.as<CallNode>();
    if (call == nullptr || OpName(call) != "tirx.address_of") {
      Fail(scope_, what + " (expected address_of, got `" + Show(arg) + "`)");
    }
    if (call->args.size() != 1) {
      Fail(scope_, what + " with " + std::to_string(call->args.size()) +
                       " arguments (expected 1)");
    }
    const auto *load = call->args[0].as<BufferLoadNode>();
    if (load == nullptr) {
      Fail(scope_, what + " (address_of of a non-buffer-load)");
    }
    if (load->indices.size() != 1) {
      Fail(scope_, what + " over " + std::to_string(load->indices.size()) +
                       " dimensions (expected 1)");
    }
    const auto *var = load->buffer->data.as<VarNode>();
    if (var == nullptr) {
      Fail(scope_, what + " (data is not a Var)");
    }
    return Ptr{var->name_hint, static_cast<std::string>(load->buffer->name),
               load->indices[0]};
  }

  // UB element index -> BYTES, refusing a non-constant base. The vector ops use
  // `i * step + base`; `base` is the constant part.
  int64_t ElemBase(const PrimExpr &ix, const std::string &what) const {
    if (auto v = ix.as<IntImmNode>()) {
      return v->value;
    }
    // `i * 64` is the slot-A form: the whole index is the per-iteration term,
    // so the constant base is 0.
    if (const auto *mul = ix.as<MulNode>()) {
      bool saw_const = false;
      for (const PrimExpr *side : {&mul->a, &mul->b}) {
        if (side->as<IntImmNode>() != nullptr) {
          saw_const = true;
        } else if (side->as<VarNode>() == nullptr) {
          Fail(scope_, what + " `" + Show(ix) + "` (non-affine term)");
        }
      }
      if (!saw_const) {
        Fail(scope_, what + " `" + Show(ix) + "` (no constant stride)");
      }
      return 0;
    }
    if (const auto *add = ix.as<AddNode>()) {
      if (auto b = add->b.as<IntImmNode>()) {
        return b->value;
      }
      if (auto a = add->a.as<IntImmNode>()) {
        return a->value;
      }
    }
    Fail(scope_, what + " `" + Show(ix) + "` (no constant base)");
  }

  // -- entry ----------------------------------------------------------------

  void EmitEntry() {
    scope_ = name_;
    Stmt body = fn_->body;
    const auto *attrs = body.as<AttrStmtNode>();
    if (attrs == nullptr || attrs->attr_key != "thread_extent") {
      Fail(name_, "entry body `" + std::string(body->GetTypeKey()) +
                      "` (expected the thread_extent AttrStmt)");
    }
    grid_ = Const(attrs->value, "grid (thread_extent value)");
    if (grid_ != kGrid) {
      Fail(scope_, "grid " + std::to_string(grid_) + " (the supported shape " +
                       "has " + std::to_string(kGrid) + " blocks)");
    }
    const auto *s = attrs->body.as<SeqStmtNode>();
    if (s == nullptr) {
      Fail(scope_, "thread_extent body as `" +
                       std::string(attrs->body->GetTypeKey()) +
                       "` (expected SeqStmt)");
    }

    // Pass 1: the arena and its aliases. They are declared before the
    // statements that use them, and the arena must be exactly one dynamic
    // allocation.
    for (const Stmt &st : s->seq) {
      if (const auto *a = st.as<AllocBufferNode>()) {
        const Buffer &b = a->buffer;
        if (b.scope() != "shared.dyn") {
          Fail(scope_, "allocation `" + b->name + "` with scope `" +
                           std::string(b.scope()) + "` (expected shared.dyn)");
        }
        if (!arena_data_.empty()) {
          Fail(scope_, "a second dynamic arena `" + b->name + "`");
        }
        const int64_t bytes =
            Const(b->shape[0], "arena size") * DtypeBytes(b->dtype);
        if (bytes != kArenaBytes) {
          Fail(scope_, "arena of " + std::to_string(bytes) +
                           " bytes (the supported shape has " +
                           std::to_string(kArenaBytes) + ")");
        }
        arena_data_ = b->data->name_hint;
      } else if (const auto *d = st.as<DeclBufferNode>()) {
        const Buffer &b = d->buffer;
        if (b.scope() != "shared.dyn") {
          Fail(scope_, "alias `" + b->name + "` with scope `" +
                           std::string(b.scope()) + "`");
        }
        alias_[static_cast<std::string>(b->name)] = b->data->name_hint;
      }
    }
    if (arena_data_.empty()) {
      Fail(scope_, "a module without a shared.dyn arena");
    }

    EmitSignature();
    EmitConstants();

    // The three UB base views the vector function consumes. They are declared
    // once, at their BYTE offsets, and every DMA and the func.call refer to the
    // SAME SSA names: a name is never minted twice for one address.
    EmitUbView(kSlotA * 4, UbName(kSlotA));
    EmitUbView(kSlotB * 4, UbName(kSlotB));
    EmitUbView(kSlotC * 4, UbName(kSlotC));

    std::string vf_fn; // collected, then emitted AFTER the entry body closes
    for (const Stmt &st : s->seq) {
      if (st->IsInstance<AllocBufferNode>() ||
          st->IsInstance<DeclBufferNode>()) {
        continue;
      }
      const auto *blk = st.as<SBlockNode>();
      if (blk == nullptr) {
        Fail(scope_, "statement `" + std::string(st->GetTypeKey()) + "`");
      }
      const std::string bname = static_cast<std::string>(blk->name_hint);
      if (bname != "VECTOR") {
        Fail(scope_, "sblock `" + bname + "`");
      }
      EmitVectorBody(GetRef<SBlock>(blk), &vf_fn);
    }

    // The completion barrier is an ENTRY responsibility (P1 D013): the Late cut
    // skips the passes that would insert it.
    Indent();
    out_ << "hivm.hir.pipe_barrier[<PIPE_ALL>]\n";
    Indent();
    out_ << "return\n";
    depth_ -= 1;
    out_ << "}\n";

    // The vector function is a SIBLING of the entry, not a nested function:
    // MLIR has no nested func.func, and the frozen Golden outlines the VF.
    if (vf_fn.empty()) {
      Fail(scope_, "a module whose VECTOR block carries no SIMD_VF");
    }
    out_ << vf_fn;
    out_ << "}\n";
  }

  // Every constant the body references, declared once at the top of the entry
  // so that a reference can never appear before its definition.
  void EmitConstants() {
    Indent();
    out_ << "%c0 = arith.constant 0 : index\n";
    Indent();
    out_ << "%c1 = arith.constant 1 : index\n";
    Indent();
    out_ << "%" << LaneStrideName() << " = arith.constant " << kLanes
         << " : index\n";
    Indent();
    out_ << "%" << BlockStrideName() << " = arith.constant "
         << kPerBlockElements << " : i64\n";
    for (int64_t slot : {kSlotA, kSlotB, kSlotC}) {
      Indent();
      out_ << "%" << UbConstName(slot) << " = arith.constant " << slot * 4
           << " : i64\n";
    }
  }

  static std::string UbName(int64_t slot) {
    return "ub_" + std::to_string(slot * 4);
  }

  // A UB byte offset that is one of the three declared slot views resolves to
  // that view's SSA name. Any other byte offset is a refusal: the frozen shape
  // has three slots and no fourth region to point at.
  std::string UbNameAt(int64_t byte_off, const std::string &what) const {
    if (byte_off == kSlotA * 4) {
      return UbName(kSlotA);
    }
    if (byte_off == kSlotB * 4) {
      return UbName(kSlotB);
    }
    if (byte_off == kSlotC * 4) {
      return UbName(kSlotC);
    }
    Fail(scope_, what + " at UB byte offset " + std::to_string(byte_off) +
                     " (the supported slots are 0 / 65536 / 131072)");
  }

  static std::string UbConstName(int64_t slot) {
    return "c_ub" + std::to_string(slot * 4) + "_";
  }

  void EmitSignature() {
    std::ostringstream params;
    for (size_t i = 0; i < fn_->params.size(); ++i) {
      const Var &p = fn_->params[i];
      // The prepared TIR holds every shape as a GM handle; the element type and
      // extent are fixed by the frozen shape and are checked here rather than
      // re-derived from an annotation.
      if (!p->dtype.is_handle()) {
        Fail("the entry signature", "parameter dtype " + Ty(p->dtype) +
                                        " (the supported shape holds GM "
                                        "handles)");
      }
      if (i != 0) {
        params << ",\n";
      }
      params << "    %" << p->name_hint << ": memref<" << kTotalElements
             << "xf32, #hivm.address_space<gm>>";
    }
    // The module attribute block. `hacc.target` and the complete
    // `dlti.target_system_spec` are not decoration: the Late pipeline's device
    // lowering reads the target to decide which instructions exist, and the
    // store's l2_cache_mode=4 is only legal once the module declares an
    // Ascend950 target. A partial spec parses cleanly and then crashes a
    // downstream pass, so the whole spec is written out.
    out_ << "module attributes {\n"
            "  hacc.target = #hacc.target<\""
         << kLateTargetName
         << "\">,\n"
            "  dlti.target_system_spec = #dlti.target_system_spec<\"NPU\" : "
            "#hacc.target_device_spec<"
            "#dlti.dl_entry<\"AI_CORE_COUNT\", 28 : i32>, "
            "#dlti.dl_entry<\"CUBE_CORE_COUNT\", 28 : i32>, "
            "#dlti.dl_entry<\"VECTOR_CORE_COUNT\", 56 : i32>, "
            "#dlti.dl_entry<\"UB_SIZE\", 2031616 : i32>, "
            "#dlti.dl_entry<\"L1_SIZE\", 4194304 : i32>, "
            "#dlti.dl_entry<\"L0A_SIZE\", 524288 : i32>, "
            "#dlti.dl_entry<\"L0B_SIZE\", 524288 : i32>, "
            "#dlti.dl_entry<\"L0C_SIZE\", 2097152 : i32>, "
            "#dlti.dl_entry<\"UB_ALIGN_SIZE\", 256 : i32>, "
            "#dlti.dl_entry<\"L1_ALIGN_SIZE\", 256 : i32>, "
            "#dlti.dl_entry<\"L0C_ALIGN_SIZE\", 4096 : i32>, "
            "#dlti.dl_entry<\"MINIMAL_D_CACHE_SIZE\", 262144 : i32>, "
            "#dlti.dl_entry<\"MAXIMUM_D_CACHE_SIZE\", 983040 : i32>, "
            "#dlti.dl_entry<\"ARCH\", \"dav-c310\">>>,\n"
            "  hivm.module_core_type = #hivm.module_core_type<AIV>\n"
            "} {\n";
    out_ << "  func.func @" << name_ << "(" << params.str() << ") "
         << "attributes {hacc.entry, "
         << "hacc.function_kind = #hacc.function_kind<DEVICE>, "
         << "hacc.block_dim = " << grid_ << " : i64, "
         << "hivm.func_core_type = #hivm.func_core_type<AIV>} {\n";
  }

  // -- the VECTOR block -----------------------------------------------------

  void EmitVectorBody(const SBlock &blk, std::string *out_fn) {
    for (const Stmt &st : IterStmts(blk->body)) {
      if (const auto *e = st.as<EvaluateNode>()) {
        EmitSideEffect(e->value);
        continue;
      }
      const auto *b = st.as<SBlockNode>();
      if (b != nullptr && static_cast<std::string>(b->name_hint) == "SIMD_VF") {
        EmitVfCall();
        *out_fn = EmitVectorFunction(GetRef<SBlock>(b));
        continue;
      }
      Fail(scope_,
           "statement `" + std::string(st->GetTypeKey()) + "` inside VECTOR");
    }
  }

  // A side effect in VECTOR: a DMA or a flag. These carry no result value, so
  // they appear as Evaluate.
  void EmitSideEffect(const PrimExpr &expr) {
    const auto *call = expr.as<CallNode>();
    if (call == nullptr) {
      Fail(scope_, "Evaluate of `" + std::string(expr->GetTypeKey()) + "`");
    }
    const std::string op = OpName(call);
    if (op == "tl.ascend_copy_gm_to_ubuf") {
      EmitGmToUb(call);
    } else if (op == "tl.ascend_copy_ubuf_to_gm") {
      EmitUbToGm(call);
    } else if (op == "tl.ascend_set_flag") {
      EmitFlag(call, "set_flag");
    } else if (op == "tl.ascend_wait_flag") {
      EmitFlag(call, "wait_flag");
    } else {
      Fail(scope_, "call `" + (op.empty() ? "<non-Op callee>" : op) + "`");
    }
  }

  // tl.ascend_copy_gm_to_ubuf(dst, src, sid, nburst, burstLen, leftPad,
  //                           rightPad, srcStride, dstStride, srcGap, dstGap)
  // burstLen / strides / gaps are BYTES; pads are ELEMENTS.
  void EmitGmToUb(const CallNode *call) {
    if (call->args.size() != 11) {
      Fail(scope_, "GM->UB copy with " + std::to_string(call->args.size()) +
                       " arguments (expected 11)");
    }
    const Ptr dst = CheckUb(AddressOf(call->args[0], "GM->UB destination"));
    const Ptr src = CheckGm(AddressOf(call->args[1], "GM->UB source"));
    const int64_t dst_elem = ElemBase(dst.index, "GM->UB UB offset");
    const int64_t dst_byte = dst_elem * DtypeBytes(DataType::Float(32));

    RequireArg(call, 2, 0, "GM->UB sid");
    RequireArg(call, 3, 1, "GM->UB burst count");
    RequireArg(call, 4, kDmaBytes, "GM->UB burst length");
    RequireArg(call, 5, 0, "GM->UB left pad");
    RequireArg(call, 6, 0, "GM->UB right pad");
    RequireArg(call, 7, 0, "GM->UB source stride");
    RequireArg(call, 8, 0, "GM->UB destination stride");
    RequireArg(call, 9, kDmaBytes, "GM->UB source gap");
    RequireArg(call, 10, kDmaBytes, "GM->UB destination gap");

    const int64_t n = kDmaBytes / DtypeBytes(DataType::Float(32));
    EmitGmSubview(src.data, src.index, n, "gm_src_" + std::to_string(dma_));
    const std::string ub = UbNameAt(dst_byte, "GM->UB destination");
    Indent();
    out_ << "hivm.hir.load ins(%gm_src_" << dma_ << " : memref<" << n
         << "xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) "
            "outs(%"
         << ub << " : memref<" << n << "xf32, #hivm.address_space<ub>>)\n";
    ++dma_;
  }

  // tl.ascend_copy_ubuf_to_gm(dst, src, sid, nburst, burstLen, cacheMode,
  //                           srcStride, dstStride)
  void EmitUbToGm(const CallNode *call) {
    if (call->args.size() != 8) {
      Fail(scope_, "UB->GM copy with " + std::to_string(call->args.size()) +
                       " arguments (expected 8)");
    }
    const Ptr dst = CheckGm(AddressOf(call->args[0], "UB->GM destination"));
    const Ptr src = CheckUb(AddressOf(call->args[1], "UB->GM source"));
    const int64_t src_elem = ElemBase(src.index, "UB->GM UB offset");
    const int64_t src_byte = src_elem * DtypeBytes(DataType::Float(32));

    // The frozen store is cache mode 4 (P1 D012). Another value is not "wrong"
    // in general, but this generator only claims the reviewed subset, so it
    // refuses rather than emit a module whose cache mode was never tested.
    const int64_t cache = Const(call->args[5], "store cache mode");
    if (cache != 4) {
      Fail(scope_, "store cache mode " + std::to_string(cache) +
                       " (the supported subset is 4)");
    }
    RequireArg(call, 2, 0, "UB->GM sid");
    RequireArg(call, 3, 1, "UB->GM burst count");
    RequireArg(call, 4, kDmaBytes, "UB->GM burst length");
    RequireArg(call, 6, kDmaBytes, "UB->GM source stride");
    RequireArg(call, 7, kDmaBytes, "UB->GM destination stride");

    const int64_t n = kDmaBytes / DtypeBytes(DataType::Float(32));
    EmitGmSubview(dst.data, dst.index, n, "gm_dst_" + std::to_string(dma_));
    const std::string ub = UbNameAt(src_byte, "UB->GM source");
    Indent();
    out_ << "hivm.hir.store ins(%" << ub << " : memref<" << n
         << "xf32, #hivm.address_space<ub>>) outs(%gm_dst_" << dma_
         << " : memref<" << n
         << "xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) "
            "{l2_cache_mode = 4 : i32}\n";
    ++dma_;
  }

  // A GM pointer names a top-level shape parameter. An unknown data var is a
  // refusal: a GM access to something that is not A / B / C would be emitted
  // against a memref this module never declares.
  Ptr CheckGm(Ptr p) {
    bool known = false;
    for (const Var &v : fn_->params) {
      if (v->name_hint == p.data) {
        known = true;
      }
    }
    if (!known) {
      Fail(scope_, "GM pointer to `" + p.data + "` (not a kernel parameter)");
    }
    return p;
  }

  Ptr CheckUb(Ptr p) {
    if (p.data != arena_data_) {
      Fail(scope_, "UB pointer to `" + p.data + "` (the arena is `" +
                       arena_data_ + "`)");
    }
    return p;
  }

  // gm[bx * per_block] -> a subview of the shape parameter.
  void EmitGmSubview(const std::string &param, const PrimExpr &ix, int64_t n,
                     const std::string &name) {
    const auto *mul = ix.as<MulNode>();
    if (mul == nullptr) {
      Fail(scope_, "GM offset `" + Show(ix) +
                       "` (expected blockIdx * per-block extent)");
    }
    int64_t stride = 0;
    bool saw_block = false;
    for (const PrimExpr *side : {&mul->a, &mul->b}) {
      if (auto v = side->as<IntImmNode>()) {
        stride = v->value;
      } else if (side->as<VarNode>() != nullptr) {
        saw_block = true;
      } else {
        Fail(scope_, "GM offset `" + Show(ix) + "` (non-affine term)");
      }
    }
    if (!saw_block) {
      Fail(scope_, "GM offset `" + Show(ix) + "` (no block index)");
    }
    if (stride != kPerBlockElements) {
      Fail(scope_, "GM block stride " + std::to_string(stride) +
                       " (this module moves " +
                       std::to_string(kPerBlockElements) +
                       " elements per block)");
    }
    Indent();
    out_ << "%" << name << "_bid = hivm.hir.get_block_idx -> i64\n";
    Indent();
    out_ << "%" << name << "_off = arith.muli %" << name << "_bid, %"
         << BlockStrideName() << " : i64\n";
    Indent();
    out_ << "%" << name << "_ix = arith.index_cast %" << name
         << "_off : i64 to index\n";
    Indent();
    out_ << "%" << name << " = memref.subview %" << param << "[%" << name
         << "_ix] [" << n << "] [1] : memref<" << kTotalElements
         << "xf32, #hivm.address_space<gm>> to memref<" << n
         << "xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>\n";
  }

  // A UB pointer_cast at a BYTE offset into the arena. The view is the tile the
  // kernel actually moves, not the whole remainder of the arena: the callee's
  // signature and the DMA extents are all kPerBlockElements wide, and a wider
  // view makes the call site disagree with the function definition.
  void EmitUbView(int64_t byte_off, const std::string &name) {
    if (byte_off < 0 || byte_off >= kArenaBytes) {
      Fail(scope_, "UB byte offset " + std::to_string(byte_off) +
                       " outside the " + std::to_string(kArenaBytes) +
                       "-byte arena");
    }
    if (byte_off % 4 != 0) {
      Fail(scope_, "UB byte offset " + std::to_string(byte_off) +
                       " (not float32-aligned)");
    }
    const int64_t elems = kPerBlockElements;
    Indent();
    out_ << "%" << name << " = hivm.hir.pointer_cast(%c_ub" << byte_off
         << "_) : memref<" << elems << "xf32, #hivm.address_space<ub>>\n";
  }

  void EmitFlag(const CallNode *call, const std::string &which) {
    if (call->args.size() != 2) {
      Fail(scope_, which + " with " + std::to_string(call->args.size()) +
                       " arguments (expected 2)");
    }
    const std::string pipe = Str(call->args[0], which + " pipe pair");
    auto it = kFlagPipes.find(pipe);
    if (it == kFlagPipes.end()) {
      Fail(scope_, which + " on pipe pair `" + pipe + "`");
    }
    const int64_t event = Const(call->args[1], which + " event id");
    if (event != 0) {
      Fail(scope_, which + " event id " + std::to_string(event) +
                       " (the supported subset is EVENT_ID0)");
    }
    Indent();
    out_ << "hivm.hir." << which << "[<" << it->second.from << ">, <"
         << it->second.to << ">, <EVENT_ID0>]\n";
  }

  void RequireArg(const CallNode *call, size_t i, int64_t want,
                  const std::string &what) {
    if (i >= call->args.size()) {
      Fail(scope_, what + " (missing argument " + std::to_string(i) + ")");
    }
    const int64_t got = Const(call->args[i], what);
    if (got != want) {
      Fail(scope_, what + " = " + std::to_string(got) + " (the supported " +
                       "subset is " + std::to_string(want) + ")");
    }
  }

  void EmitVfCall() {
    Indent();
    out_ << "func.call @" << name_ << "_simd_vf_0(%" << UbName(kSlotA) << ", %"
         << UbName(kSlotB) << ", %" << UbName(kSlotC)
         << ") {hivm.vector_function, no_inline} : "
            "(memref<"
         << kPerBlockElements
         << "xf32, #hivm.address_space<ub>>, "
            "memref<"
         << kPerBlockElements
         << "xf32, #hivm.address_space<ub>>, "
            "memref<"
         << kPerBlockElements << "xf32, #hivm.address_space<ub>>) -> ()\n";
  }

  // -- the outlined vector function ----------------------------------------

  std::string EmitVectorFunction(const SBlock &vf) {
    std::ostringstream saved;
    saved << out_.str();
    const int saved_depth = depth_;
    out_.str("");
    out_.clear();
    out_ << "\n  func.func @" << name_ << "_simd_vf_0(\n"
         << "      %a: memref<" << kPerBlockElements
         << "xf32, #hivm.address_space<ub>>,\n"
         << "      %b: memref<" << kPerBlockElements
         << "xf32, #hivm.address_space<ub>>,\n"
         << "      %c: memref<" << kPerBlockElements
         << "xf32, #hivm.address_space<ub>>) "
            "attributes {hivm.func_core_type = "
            "#hivm.func_core_type<AIV>, "
            "hivm.vector_function, no_inline} {\n";
    depth_ = 2;
    Indent();
    out_ << "%c0 = arith.constant 0 : index\n";
    Indent();
    out_ << "%c1 = arith.constant 1 : index\n";
    // The lane stride is a fresh constant inside this function: it is defined
    // in the entry function's scope, which an outlined function cannot
    // reference.
    Indent();
    out_ << "%" << LaneStrideName() << " = arith.constant " << kLanes
         << " : index\n";

    for (const Stmt &st : UnwrapScopes(vf->body)) {
      if (const auto *lp = st.as<ForNode>()) {
        EmitVfLoop(GetRef<For>(lp));
        continue;
      }
      // `mask: T.boolx256 = T.tl.simd.pset(...)` is a Bind of the predicate
      // register, so it is consumed here exactly like the loop-body Binds.
      if (const auto *bind = st.as<BindNode>()) {
        const auto *bc = bind->value.as<CallNode>();
        const std::string bop = bc == nullptr ? std::string() : OpName(bc);
        if (bop != kOpPset) {
          Fail(scope_, "binding of `" + (bop.empty() ? "<non-Op>" : bop) +
                           "` outside the VF loop (only pset is supported)");
        }
        EmitPset(bc);
        continue;
      }
      const auto *e = st.as<EvaluateNode>();
      if (e == nullptr) {
        Fail(scope_, "statement `" + std::string(st->GetTypeKey()) +
                         "` inside SIMD_VF");
      }
      const std::string op = OpName(e->value.as<CallNode>());
      if (op == kOpPset) {
        EmitPset(e->value.as<CallNode>());
      } else {
        Fail(scope_, "call `" + (op.empty() ? "<non-Op>" : op) +
                         "` outside the VF loop");
      }
    }
    Indent();
    out_ << "return\n  }\n";
    depth_ = saved_depth;
    std::string text = out_.str();
    out_.str("");
    out_.clear();
    out_ << saved.str();
    return text;
  }

  void EmitPset(const CallNode *call) {
    const int64_t lanes = Const(call->args[0], "pset pattern length");
    if (lanes != 32) {
      Fail(scope_, "pset pattern length " + std::to_string(lanes) +
                       " (the supported subset is 32)");
    }
    const std::string pat = Str(call->args[1], "pset pattern");
    if (pat != "PAT_ALL") {
      Fail(scope_, "pset pattern `" + pat + "` (the supported subset is ALL)");
    }
    Indent();
    out_ << "%mask = ave.hir.pge <ALL> {functionType = "
            "#ave.func_dist_type<pb32>} "
            ": vector<"
         << kMaskLanes << "xi1>\n";
  }

  void EmitVfLoop(const For &loop) {
    if (loop->kind != ForKind::kSerial) {
      Fail(scope_,
           "VECTOR loop kind " + std::to_string(static_cast<int>(loop->kind)));
    }
    const int64_t extent = Const(loop->extent, "VF loop extent");
    const int64_t start = Const(loop->min, "VF loop min");
    if (start != 0) {
      Fail(scope_, "VF loop min " + std::to_string(start) +
                       " (the supported subset is 0)");
    }
    loop_var_ = loop->loop_var->name_hint;

    // The TIR iterates 128 times; the Late form iterates over ELEMENTS. 128 x
    // 64 lanes is the 8192-element tile, so the bound is extent*kLanes and the
    // step is the lane count. Either number alone would change which elements
    // each iteration touches.
    const int64_t elements = extent * kLanes;
    Indent();
    out_ << "%c" << elements << "_elem = arith.constant " << elements
         << " : index\n";
    Indent();
    out_ << "scf.for %" << loop_var_ << " = %c0 to %c" << elements
         << "_elem step %" << LaneStrideName() << " {\n";
    depth_ += 1;

    for (const Stmt &st : IterStmts(loop->body)) {
      // `tirx.Bind` binds the vector temporary the next expression consumes;
      // the bound value IS the computation, so it is emitted in place.
      if (const auto *bind = st.as<BindNode>()) {
        EmitVfValue(bind->value);
        continue;
      }
      const auto *e = st.as<EvaluateNode>();
      if (e == nullptr) {
        Fail(scope_, "statement `" + std::string(st->GetTypeKey()) +
                         "` in the VF loop");
      }
      EmitVfValue(e->value);
    }

    depth_ -= 1;
    Indent();
    out_ << "}\n";
    loop_var_.clear();
  }

  // A vector value: vld (produces a register), vadd (consumes two), vsts
  // (consumes one). Registers are tracked as a stack in program order, which is
  // the SSA order the emitted text uses.
  void EmitVfValue(const PrimExpr &expr) {
    const auto *call = expr.as<CallNode>();
    if (call == nullptr) {
      Fail(scope_, "vector value `" + std::string(expr->GetTypeKey()) + "`");
    }
    const std::string op = OpName(call);
    if (op == kOpVld) {
      EmitVld(call);
    } else if (op == kOpVadd) {
      EmitVadd(call);
    } else if (op == kOpVsts) {
      EmitVsts(call);
    } else if (op == kOpPset) {
      EmitPset(call);
    } else {
      Fail(scope_, "call `" + (op.empty() ? "<non-Op>" : op) + "` in the VF");
    }
  }

  void EmitVld(const CallNode *call) {
    if (call->args.size() != 2) {
      Fail(scope_, "vld with " + std::to_string(call->args.size()) +
                       " arguments (expected 2)");
    }
    const Ptr src = CheckUb(AddressOf(call->args[0], "vld address"));
    const std::string mode = Str(call->args[1], "vld mode");
    if (mode != "NORM") {
      Fail(scope_, "vld mode `" + mode + "` (the supported subset is NORM)");
    }
    const int64_t byte_off =
        ElemBase(src.index, "vld UB offset") * DtypeBytes(DataType::Float(32));
    const std::string view = EmitVfSlotView(byte_off, "vld");
    const std::string reg = "v" + std::to_string(reg_count_);
    Indent();
    out_ << "%" << reg << " = ave.hir.vload <NORM> %" << view
         << "[%c0] : memref<" << kLanes
         << "xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into "
            "vector<"
         << kLanes << "xf32>\n";
    regs_.push_back(reg);
    ++reg_count_;
  }

  void EmitVadd(const CallNode *call) {
    if (call->args.size() != 4) {
      Fail(scope_, "vadd with " + std::to_string(call->args.size()) +
                       " arguments (expected 4)");
    }
    if (regs_.size() < 2) {
      Fail(scope_, "vadd without two vector operands in scope");
    }
    const std::string rhs = regs_.back();
    regs_.pop_back();
    const std::string lhs = regs_.back();
    regs_.pop_back();
    const std::string mode = Str(call->args[3], "vadd mask mode");
    if (mode != "MODE_ZEROING") {
      Fail(scope_, "vadd mask mode `" + mode +
                       "` (the supported subset is MODE_ZEROING)");
    }
    const std::string reg = "v" + std::to_string(reg_count_);
    Indent();
    out_ << "%" << reg << " = ave.hir.vadd %" << lhs << ", %" << rhs
         << ", %mask : vector<" << kLanes << "xf32>, vector<" << kMaskLanes
         << "xi1>\n";
    regs_.push_back(reg);
    ++reg_count_;
  }

  void EmitVsts(const CallNode *call) {
    if (call->args.size() != 4) {
      Fail(scope_, "vsts with " + std::to_string(call->args.size()) +
                       " arguments (expected 4)");
    }
    if (regs_.empty()) {
      Fail(scope_, "vsts without a vector operand in scope");
    }
    const std::string val = regs_.back();
    regs_.pop_back();
    const Ptr dst = CheckUb(AddressOf(call->args[0], "vsts address"));
    const std::string mode = Str(call->args[3], "vsts store mode");
    if (mode != "NORM_B32") {
      Fail(scope_,
           "vsts mode `" + mode + "` (the supported subset is NORM_B32)");
    }
    const int64_t byte_off =
        ElemBase(dst.index, "vsts UB offset") * DtypeBytes(DataType::Float(32));
    const std::string view = EmitVfSlotView(byte_off, "vsts");
    Indent();
    out_ << "ave.hir.masked_store <NORM_B32> %" << view << "[%c0], %mask, %"
         << val << " : memref<" << kLanes
         << "xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, "
         << "vector<" << kMaskLanes << "xi1>, vector<" << kLanes << "xf32>\n";
  }

  // A UB access inside the VF becomes a subview of the OUTLINED FUNCTION
  // argument. Which argument an offset belongs to is decided by the offset, not
  // by an alias name, because all three Golden aliases share one data var.
  std::string EmitVfSlotView(int64_t byte_off, const std::string &what) {
    int64_t slot_byte = -1;
    std::string arg;
    if (byte_off == kSlotA * 4) {
      slot_byte = kSlotA * 4;
      arg = "a";
    } else if (byte_off == kSlotB * 4) {
      slot_byte = kSlotB * 4;
      arg = "b";
    } else if (byte_off == kSlotC * 4) {
      slot_byte = kSlotC * 4;
      arg = "c";
    }
    if (slot_byte < 0) {
      Fail(scope_, what + " UB byte offset " + std::to_string(byte_off) +
                       " (the supported subset is 0 / 65536 / 131072)");
    }
    // The loop variable IS the element offset: the loop runs 0 .. 8192 step 64.
    // The constant part of the TIR offset has already been folded into which
    // argument this is, so the emitted index is the loop variable alone. There
    // is deliberately no `arith.muli` here: the block-offset validation reads
    // every `muli` in the module as a GM block scale, and a lane stride written
    // that way is misread as one.
    const std::string idx = std::to_string(view_count_);
    Indent();
    out_ << "%" << what << "_" << idx << " = memref.subview %" << arg << "[%"
         << loop_var_ << "] [" << kLanes << "] [1] : memref<"
         << kPerBlockElements << "xf32, #hivm.address_space<ub>> to memref<"
         << kLanes
         << "xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>\n";
    ++view_count_;
    return what + "_" + idx;
  }

  // `T.attr("simdvf", "tl.simdvf_scope", 1)` wraps the VF body in an AttrStmt.
  // It is a scope marker with no hardware meaning of its own: the AscendC
  // codegen treats it as transparent and only visits the body. It is unwrapped
  // here, and any OTHER AttrStmt is still refused, so the transparency cannot
  // silently swallow an annotation that does carry meaning.
  std::vector<Stmt> UnwrapScopes(const Stmt &body) {
    if (const auto *attr = body.as<AttrStmtNode>()) {
      if (attr->attr_key != "tl.simdvf_scope") {
        Fail(scope_, "AttrStmt `" + attr->attr_key +
                         "` (only the transparent simdvf scope marker is "
                         "supported)");
      }
      return UnwrapScopes(attr->body);
    }
    return IterStmts(body);
  }

  std::vector<Stmt> IterStmts(const Stmt &body) {
    if (const auto *s = body.as<SeqStmtNode>()) {
      return std::vector<Stmt>(s->seq.begin(), s->seq.end());
    }
    return {body};
  }

  PrimFunc fn_;
  std::string name_;
  std::string scope_ = "the module";
  std::string arena_data_;
  std::unordered_map<std::string, std::string> alias_;
  int64_t grid_ = 1;
  std::ostringstream out_;
  int depth_ = 2;
  int64_t dma_ = 0;
  int64_t reg_count_ = 0;
  int64_t view_count_ = 0;
  std::string loop_var_;
  std::vector<std::string> regs_;
};

std::string EmitLateMlir(const tirx::PrimFunc &func) {
  if (!func.defined()) {
    throw Unsupported("HIVM CodeGen: unsupported an undefined PrimFunc");
  }
  return Emitter(func).Run();
}

} // namespace hivm

} // namespace codegen
} // namespace tvm
