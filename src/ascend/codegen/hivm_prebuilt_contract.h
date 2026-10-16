/*!
 * \file hivm_prebuilt_contract.h
 * \brief TEST_ONLY fixed-program contract guard for the PREBUILT HIVM path.
 *
 * WHY THIS EXISTS
 * ExtractFuncInfo copies whatever the prepared device TIR declares: any symbol,
 * any number of parameters, any dtypes, any launch tags. The generic runtime
 * then packs whatever the FunctionInfo says. So a correct, immutable device
 * binary can be paired with incompatible prepared TIR -- a wrong symbol, a
 * scalar where a GM pointer belongs, swapped arguments, missing or reordered
 * launch tags, a wrong grid or a wrong dynamic-UB size -- and everything stays
 * internally consistent while the host ABI handed to the loader is wrong.
 * BC-2 requires this guard in ADDITION to ExtractFuncInfo, before the module is
 * created.
 *
 * THE CONTRACT IS DERIVED FROM THE FROZEN FIXTURE, NOT FROM THIS CODE.
 * Every expected value below was measured from
 * tests/golden/hivm/vadd_f32_single_tile/golden_codegen_input.json (and its
 * .tir sibling) by the probe recorded at
 * hivm-poc-evidence/p4-20261016/impl/p4b-bc2-guard/derive_contract.py. Deriving
 * the contract from ExtractFuncInfo -- the code under test -- would measure the
 * design against itself.
 *
 * A/B ORDER CANNOT BE VALIDATED NUMERICALLY. VADD commutes, so no numerical
 * comparison distinguishes A from B. The order evidence used here is structural:
 *   1. tl.readonly_param_indices == [0, 1] marks A and B as read and C as
 *      written, which is what fixes C's position independently of the copies.
 *   2. The GM<->UB copies appear in program order A-copy, B-copy, C-copy, and
 *      the read-only prefix of that order must agree with (1).
 * Together they are a typed-IR dataflow statement, not a numerical one.
 *
 * POINTER TYPES ARE CHECKED BEFORE REDUCTION. FunctionInfo only carries
 * DLDataType, and a handle parameter reduces to `handle` -- the element type
 * (`float32`) and address space (`global`) are lost. So the checks below read
 * PrimFunc params' PointerType annotations directly, before ExtractFuncInfo
 * flattens them.
 *
 * SCOPE: this guard is deliberately narrow and applies ONLY to the PREBUILT
 * HIVM entry. It must not run on the AscendC or PTO paths, whose programs are
 * not this fixed program. It is a TEST_ONLY contract, not a general validator.
 */

#ifndef TVM_TL_ASCEND_CODEGEN_HIVM_PREBUILT_CONTRACT_H_
#define TVM_TL_ASCEND_CODEGEN_HIVM_PREBUILT_CONTRACT_H_

#include <tvm/ffi/string.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace codegen {
namespace hivm_prebuilt {

// ---------------------------------------------------------------------------
// The fixed program, measured from the frozen Golden fixture.
// ---------------------------------------------------------------------------
struct FixedProgramContract {
  static constexpr const char *kSymbol = "main_kernel";
  static constexpr int kNumPointerParams = 3;
  static constexpr const char *kParamNames[3] = {"A", "B", "C"};
  static constexpr const char *kElementType = "float32";
  static constexpr const char *kStorageScope = "global";
  static constexpr const char *kFirstTag = "blockIdx.x";
  static constexpr const char *kLastTag = "tirx.use_dyn_shared_memory";
  static constexpr int64_t kGrid = 64;
  static constexpr int64_t kDynUbBytes = 196608;
  // HostArgs size for three 8-byte pointers. Matches __CCE_KernelArgSize of the
  // frozen binary (P3 CR-ELF-8) and the packed buffer the runtime builds.
  static constexpr size_t kHostArgsBytes = 24;
  // Read-only parameter positions; C (index 2) is written.
  static constexpr int kReadOnlyParamIndices[2] = {0, 1};
};

namespace detail {

inline std::string ToStd(const ffi::String &s) {
  return static_cast<std::string>(s);
}

// Collect the data variable names each GM<->UB copy touches, in program order.
// Returns one entry per copy call: "A", "B" or "C".
inline std::vector<std::string> CollectCopyOrder(const tirx::PrimFuncNode *f) {
  std::vector<std::string> order;
  tirx::PostOrderVisit(f->body, [&](const ffi::ObjectRef &node) {
    if (const auto *call = node.as<tirx::CallNode>()) {
      const auto *op = call->op.as<tvm::OpNode>();
      const std::string op_name = op != nullptr ? ToStd(op->name) : std::string();
      const bool is_gm_to_ub = op_name == "tl.ascend_copy_gm_to_ubuf";
      const bool is_ub_to_gm = op_name == "tl.ascend_copy_ubuf_to_gm";
      if (!is_gm_to_ub && !is_ub_to_gm) {
        return;
      }
      // The GM-side operand is arg1 for gm->ub and arg0 for ub->gm. Walk that
      // operand for a reference to one of the named parameters.
      //
      // NOTE: PostOrderVisit does NOT descend into a Buffer's data Var -- the
      // Var is reached through the BufferLoad's `buffer` field, not as a child
      // expression. So the parameter is identified from BufferLoadNode::buffer
      // (and from a bare Var, for the access_ptr form). A walker that only
      // matched VarNode sees the loop variable `bx` and nothing else, which is
      // how the first version of this guard silently failed to identify A/B/C.
      const auto &gm_side = is_gm_to_ub ? call->args[1] : call->args[0];
      std::string found;
      tirx::PostOrderVisit(gm_side, [&](const ffi::ObjectRef &inner) {
        const tirx::VarNode *var = nullptr;
        if (const auto *bl = inner.as<tirx::BufferLoadNode>()) {
          var = bl->buffer->data.as<tirx::VarNode>();
        } else if (const auto *v = inner.as<tirx::VarNode>()) {
          var = v;
        }
        if (var == nullptr) {
          return;
        }
        const std::string n = ToStd(var->name_hint);
        for (const char *p : FixedProgramContract::kParamNames) {
          if (n == p) {
            found = n;
          }
        }
      });
      order.push_back(found.empty() ? std::string("?") : found);
    }
  });
  return order;
}

} // namespace detail

/*!
 * \brief Validate the prepared device IR against the fixed program contract.
 *
 * Throws (TVM_FFI_ICHECK) with a named cause on the FIRST violation, so a
 * mutation produces a specific refusal rather than a generic failure. Called
 * BEFORE the module is created.
 */
inline void CheckFixedProgramContract(const IRModule &mod,
                                      const Target &target) {
  using detail::ToStd;
  using FPC = FixedProgramContract;
  constexpr const char *kSymbol = FPC::kSymbol;
  constexpr int kNumPointerParams = FPC::kNumPointerParams;
  constexpr const char *kElementType = FPC::kElementType;
  constexpr const char *kStorageScope = FPC::kStorageScope;
  constexpr const char *kFirstTag = FPC::kFirstTag;
  constexpr const char *kLastTag = FPC::kLastTag;
  constexpr int64_t kGrid = FPC::kGrid;
  constexpr int64_t kDynUbBytes = FPC::kDynUbBytes;
  // The PREBUILT path must be reached through an explicit HIVM selection, not
  // through a plain AscendC target that happens to flow here.
  TVM_FFI_ICHECK(target.defined())
      << "HIVM PREBUILT contract: target is undefined";
  TVM_FFI_ICHECK(target->kind->name == "ascend")
      << "HIVM PREBUILT contract: expected target kind 'ascend', got '"
      << target->kind->name << "'";
  bool has_hivm_key = false;
  for (const auto &k : target->keys) {
    if (ToStd(k) == "hivm") {
      has_hivm_key = true;
    }
  }
  TVM_FFI_ICHECK(has_hivm_key)
      << "HIVM PREBUILT contract: target keys must carry the explicit 'hivm' "
         "key; got ["
      << [&] {
           std::string s;
           for (const auto &k : target->keys) {
             if (!s.empty()) s += ", ";
             s += ToStd(k);
           }
           return s;
         }()
      << "]";

  // --- exactly one device kernel -------------------------------------------
  std::vector<const tirx::PrimFuncNode *> kernels;
  for (auto kv : mod->functions) {
    const auto *f = kv.second.as<tirx::PrimFuncNode>();
    if (f == nullptr) {
      continue;
    }
    auto cc = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    if (cc.has_value() && cc.value() == CallingConv::kDeviceKernelLaunch) {
      kernels.push_back(f);
    }
  }
  TVM_FFI_ICHECK(kernels.size() == 1)
      << "HIVM PREBUILT contract: expected exactly 1 device kernel, got "
      << kernels.size();

  const tirx::PrimFuncNode *f = kernels[0];

  // --- single entry symbol -------------------------------------------------
  auto sym_opt = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  TVM_FFI_ICHECK(sym_opt.has_value())
      << "HIVM PREBUILT contract: device kernel has no global_symbol";
  const std::string sym = ToStd(sym_opt.value());
  TVM_FFI_ICHECK(sym == kSymbol)
      << "HIVM PREBUILT contract: expected entry symbol '" << kSymbol
      << "', got '" << sym << "'";

  // --- three pointer params, their ORDER, types and address spaces ---------
  TVM_FFI_ICHECK(f->params.size() == static_cast<size_t>(kNumPointerParams))
      << "HIVM PREBUILT contract: expected " << kNumPointerParams
      << " parameters, got " << f->params.size();

  for (int i = 0; i < kNumPointerParams; ++i) {
    const auto &p = f->params[i];
    // Check the PointerType BEFORE ExtractFuncInfo reduces it to `handle`.
    const auto *ptr = p->type_annotation.as<PointerTypeNode>();
    TVM_FFI_ICHECK(ptr != nullptr)
        << "HIVM PREBUILT contract: parameter " << i << " ('" << ToStd(p->name_hint)
        << "') is not a pointer; a scalar here would be packed as a scalar";
    const auto *prim = ptr->element_type.as<PrimTypeNode>();
    TVM_FFI_ICHECK(prim != nullptr)
        << "HIVM PREBUILT contract: parameter " << i
        << " pointer element type is not a primitive type";
    std::ostringstream elem_ss;
    elem_ss << prim->dtype;
    const std::string elem = elem_ss.str();
    TVM_FFI_ICHECK(elem == kElementType)
        << "HIVM PREBUILT contract: parameter " << i << " element type must be '"
        << kElementType << "', got '" << elem << "'";
    const std::string scope = ToStd(ptr->storage_scope);
    TVM_FFI_ICHECK(scope == kStorageScope)
        << "HIVM PREBUILT contract: parameter " << i << " address space must be '"
        << kStorageScope << "', got '" << scope << "'";
    TVM_FFI_ICHECK(ToStd(p->name_hint) == FPC::kParamNames[i])
        << "HIVM PREBUILT contract: parameter " << i << " must be '"
        << FPC::kParamNames[i] << "', got '" << ToStd(p->name_hint) << "'";
  }

  // --- exact launch tags AND their order ----------------------------------
  auto tags_opt =
      f->GetAttr<ffi::Array<ffi::String>>(tirx::attr::kKernelLaunchParams);
  TVM_FFI_ICHECK(tags_opt.has_value())
      << "HIVM PREBUILT contract: device kernel has no kernel_launch_params";
  const auto &tags = tags_opt.value();
  TVM_FFI_ICHECK(tags.size() == 2)
      << "HIVM PREBUILT contract: expected 2 launch tags, got " << tags.size();
  TVM_FFI_ICHECK(ToStd(tags[0]) == kFirstTag)
      << "HIVM PREBUILT contract: launch tag 0 must be '" << kFirstTag
      << "', got '" << ToStd(tags[0]) << "'";
  TVM_FFI_ICHECK(ToStd(tags[1]) == kLastTag)
      << "HIVM PREBUILT contract: launch tag 1 (last) must be '" << kLastTag
      << "', got '" << ToStd(tags[1]) << "'";

  // --- grid and dynamic UB -------------------------------------------------
  // These two attributes have no tvm::attr constant; the rest of the codebase
  // reads them by literal string (see src/transform/lower_device_kernel_launch.cc
  // :298-306 and src/metal/codegen/codegen_metal.cc:151), so the same literals
  // are used here.
  auto te = f->GetAttr<ffi::Map<ffi::String, PrimExpr>>("thread_extent");
  TVM_FFI_ICHECK(te.has_value())
      << "HIVM PREBUILT contract: device kernel has no thread_extent";
  TVM_FFI_ICHECK(te.value().size() == 1)
      << "HIVM PREBUILT contract: expected exactly 1 thread extent";
  const auto &extent_kv = *te.value().begin();
  TVM_FFI_ICHECK(ToStd(extent_kv.first) == kFirstTag)
      << "HIVM PREBUILT contract: thread extent must be on '" << kFirstTag
      << "', got '" << ToStd(extent_kv.first) << "'";
  const auto *grid = extent_kv.second.as<IntImmNode>();
  TVM_FFI_ICHECK(grid != nullptr)
      << "HIVM PREBUILT contract: thread extent is not a constant integer";
  TVM_FFI_ICHECK(grid->value == kGrid)
      << "HIVM PREBUILT contract: grid must be " << kGrid << ", got "
      << grid->value;

  auto dub = f->GetAttr<IntImm>("dyn_shared_memory_buf");
  TVM_FFI_ICHECK(dub.has_value())
      << "HIVM PREBUILT contract: device kernel has no dyn_shared_memory_buf";
  TVM_FFI_ICHECK(dub.value()->value == kDynUbBytes)
      << "HIVM PREBUILT contract: dynamic UB must be " << kDynUbBytes
      << " bytes, got " << dub.value()->value;

  // --- A/B order, structurally --------------------------------------------
  // (1) The read-only parameter positions must be exactly A and B.
  auto ro = f->GetAttr<ffi::Array<Integer>>(
      ffi::String("tl.readonly_param_indices"));
  TVM_FFI_ICHECK(ro.has_value())
      << "HIVM PREBUILT contract: device kernel has no "
         "tl.readonly_param_indices, so A/B/C order cannot be established";
  TVM_FFI_ICHECK(ro.value().size() == 2)
      << "HIVM PREBUILT contract: expected 2 read-only params (A, B), got "
      << ro.value().size();
  for (int i = 0; i < 2; ++i) {
    TVM_FFI_ICHECK(ro.value()[i]->value == FPC::kReadOnlyParamIndices[i])
        << "HIVM PREBUILT contract: read-only param " << i << " must be index "
        << FPC::kReadOnlyParamIndices[i] << ", got " << ro.value()[i]->value;
  }
  // (2) The GM<->UB copy order must be A, B, then C.
  const std::vector<std::string> copy_order = detail::CollectCopyOrder(f);
  TVM_FFI_ICHECK(copy_order.size() == 3)
      << "HIVM PREBUILT contract: expected 3 GM<->UB copies (A, B, C), got "
      << copy_order.size();
  for (int i = 0; i < 3; ++i) {
    TVM_FFI_ICHECK(copy_order[i] == FPC::kParamNames[i])
        << "HIVM PREBUILT contract: GM<->UB copy " << i << " must touch '"
        << FPC::kParamNames[i] << "', got '" << copy_order[i]
        << "' -- an A/B swap is invisible numerically and is caught here";
  }
}

} // namespace hivm_prebuilt
} // namespace codegen
} // namespace tvm

#endif // TVM_TL_ASCEND_CODEGEN_HIVM_PREBUILT_CONTRACT_H_
