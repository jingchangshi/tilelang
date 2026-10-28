/*!
 * \file rt_mod_ascend_hivm.cc
 * \brief The real (non-PREBUILT) HIVM build entry.
 *
 * TWO ENTRIES, ON PURPOSE
 *   `target.build.tilelang_hivm`          this file: generate MLIR, compile it
 *                                         with the P2 Late pipeline, create a
 *                                         module from the NEW binary.
 *   `target.build.tilelang_hivm_prebuilt` rt_mod_ascend.cc: TEST_ONLY, loads
 * the frozen P3 program. Kept, and kept isolated, so a regression can still
 *                                         exercise the P4 runtime path.
 *
 * WHAT IS NOT HERE
 *   * No frozen binary is read, and there is no fallback to one. A failed
 *     generation or a failed compile is an error that carries its reason.
 *   * No AscendC source codegen, no PTO codegen.
 *   * No caching. goal P5 section 6.3 keeps the HIVM kernel cache disabled; a
 *     cache hit would return a module whose binary this round never produced.
 *
 * THE FORMAT TOKEN
 * The module is created with `kHivmGeneratedFormat`, a distinct token that
 * shares the VECTOR_CORE loader policy but is NOT the PREBUILT token. That
 * separation is what stops a generated module from describing itself, after a
 * save/restore round trip, as the frozen P3 artefact (ascend_module.cc
 * InspectSource reconstructs the PREBUILT label only for FormatIsPrebuilt).
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>

#include "ascend/codegen/hivm_codegen.h"
#include "ascend/codegen/sha256.h"
#include "ascend/hivm_prebuilt_identity.h"
#include "runtime/metadata.h"

#include <sstream>
#include <string>
#include <vector>

// CSourceModuleCreate: the existing source-module constructor the AscendC
// source-only entry (rt_mod_ascend.cc) already uses. Declared here rather than
// included, because `target/source/codegen_source_base.h` pulls in the TVM
// source-codegen headers this translation unit otherwise does not need.
namespace tvm {
namespace codegen {
ffi::Module CSourceModuleCreate(const ffi::String &code, const ffi::String &fmt,
                                const ffi::Array<ffi::String> &func_names,
                                const ffi::Array<ffi::String> &const_vars = {});
} // namespace codegen
} // namespace tvm

namespace tvm {

// Defined in src/ascend/runtime/ascend_module.cc.
namespace ascend {
ffi::Module
AscendModuleCreate(ffi::Bytes code, ffi::String fmt,
                   ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                   ffi::Map<ffi::String, ffi::String> source);
} // namespace ascend

namespace codegen {

namespace {

using ffi::GetRef;

// The compiler hook. It is looked up by name so that this file links and runs
// even when no compiler callback is registered: the Source-only entry
// (`target.build.tilelang_hivm_source_only`) must work on a host with no
// Bisheng at all, and this entry must then fail with a REASON rather than at
// load time.
constexpr const char *kCompileHook = "tilelang_callback_hivm_compile";
constexpr const char *kSourceOnlyHook =
    "target.build.tilelang_hivm_source_only";
constexpr const char *kSourceOnlyModuleHook =
    "target.build.tilelang_hivm_source_only_module";
// The `fmt` tag of a source-only module. `source_module.cc` keys its write
// behaviour on the tag, so a distinct one keeps the generated text from being
// mistaken for AscendC C source by anything that inspects the format.
constexpr const char *kSourceOnlyFormat = "late-hivm-mlir";

// The public entry selection. A plain `ascend` target reaching this file is a
// routing error, and it is refused before any work is done.
void CheckHivmTarget(const Target &target) {
  TVM_FFI_ICHECK(target.defined()) << "HIVM CODEGEN: target is undefined";
  TVM_FFI_ICHECK(target->kind->name == "ascend")
      << "HIVM CODEGEN: expected target kind 'ascend', got '"
      << target->kind->name << "'";
  bool has_hivm = false;
  for (const auto &k : target->keys) {
    if (static_cast<std::string>(k) == "hivm") {
      has_hivm = true;
    }
  }
  TVM_FFI_ICHECK(has_hivm)
      << "HIVM CODEGEN: the target must carry the explicit 'hivm' key; "
         "a plain Ascend/PTO target must not reach this entry";
}

// The emitted program and the ONE symbol both the `func.func` name and the
// runtime FunctionInfo use. They are produced together, from the same
// resolution, because deriving them separately is what allowed an emitted
// `main_kernel` to be keyed as something else in the function map.
struct Emitted {
  std::string mlir;
  std::string symbol;
};

// Emit the MLIR for the ONE device kernel the module carries. A module with a
// different number of device kernels is not silently reduced to its first one.
Emitted EmitModule(const IRModule &mod, tirx::PrimFunc *out_func) {
  std::vector<std::pair<GlobalVar, tirx::PrimFunc>> kernels;
  for (const auto &kv : mod->functions) {
    if (auto f = kv.second.as<tirx::PrimFuncNode>()) {
      kernels.emplace_back(Downcast<GlobalVar>(kv.first),
                           GetRef<tirx::PrimFunc>(f));
    }
  }
  TVM_FFI_ICHECK_EQ(kernels.size(), 1)
      << "HIVM CODEGEN: exactly one device PrimFunc is supported, got "
      << kernels.size();
  *out_func = kernels[0].second;
  const std::string symbol = hivm::ResolveKernelSymbol(
      kernels[0].second,
      static_cast<std::string>(kernels[0].first->name_hint));
  return Emitted{hivm::EmitLateMlir(kernels[0].second, symbol), symbol};
}

// The per-kernel FunctionInfo the runtime needs for the launch grid and the
// argument packing. It is extracted from the ORIGINAL prepared TIR, not from
// the generated text: the TIR is the authority on the ABI, and deriving it
// twice from two different artefacts is how they drift apart.
ffi::Map<ffi::String, runtime::FunctionInfo>
ExtractGeneratedFuncInfo(const IRModule &mod, const std::string &symbol) {
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap;
  for (const auto &kv : mod->functions) {
    auto f = Downcast<tirx::PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    if (!(calling_conv == CallingConv::kDeviceKernelLaunch)) {
      continue;
    }
    ffi::Array<DLDataType> arg_types;
    for (const auto &p : f->params) {
      DataType dtype = p.dtype();
      if (dtype.is_bool()) {
        dtype = DataType::Int(32);
      }
      arg_types.push_back(dtype);
    }
    ffi::Array<ffi::String> launch_param_tags;
    if (auto opt = f->GetAttr<ffi::Array<ffi::String>>(
            tirx::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        launch_param_tags.push_back(tag);
      }
    }
    // Keyed by the SAME resolved symbol the emitted `func.func` carries. This
    // used to re-derive the name here, so a function with no `global_symbol`
    // emitted `main_kernel` while the metadata was keyed by the GlobalVar name.
    fmap.Set(ffi::String(symbol), runtime::FunctionInfo(ffi::String(symbol),
                                                        arg_types,
                                                        launch_param_tags, {}));
  }
  return fmap;
}

// The identity of one generation: the MLIR text this module was generated from.
// It is carried in the module's source map so the GENERATED artefact has a
// traceable relationship to its INPUT, not merely a descriptive label. Nothing
// here is a digest of the frozen P3 binary; the sha256 is of the text THIS
// process just emitted.
std::string GeneratedProvenance(const std::string &mlir,
                                const std::string &symbol) {
  const std::string mlir_digest = hivm_prebuilt::Sha256Hex(mlir);
  std::ostringstream label;
  label << "// GENERATED device binary by the TileLang HIVM DeviceCodeGen.\n"
           "// Path: prepared TIR -> HIVM TIR visitor -> Late HIVM/AVE MLIR -> "
           "P2 Late pipeline.\n"
           "// Kernel symbol: "
        << symbol << "\n"
           "// Input MLIR: "
        << mlir.size() << " bytes, sha256 " << mlir_digest
        << "\n"
           "// Compiler identity: NOT recorded here. This label ties the binary "
           "to the\n"
           "// MLIR it came from; it does NOT say which compiler produced it. "
           "The\n"
           "// compiler is configuration (HIVM_COMPILER), measured on demand by\n"
           "// tilelang.contrib.hivm.compiler_identity(), and a build that "
           "needs that\n"
           "// attribution must record it -- it is deliberately not asserted "
           "here,\n"
           "// because this code path never receives it.\n"
           "// The trailer below IS the Late MLIR this binary was produced "
           "from.\n"
           "// It is source, not a claim: a reader can compare it against the "
           "binary.\n";
  label << "// --- BEGIN GENERATED LATE MLIR ---\n" << mlir;
  if (!mlir.empty() && mlir.back() != '\n') {
    label << "\n";
  }
  label << "// --- END GENERATED LATE MLIR ---\n";
  return label.str();
}

// SOURCE-ONLY: return the generated MLIR text and nothing else. No compiler is
// invoked and no device is needed, so this is the entry the P5 Gate uses for
// the semantics tests.
ffi::String BuildHivmSourceOnly(IRModule mod, Target target) {
  CheckHivmTarget(target);
  tirx::PrimFunc kernel;
  Emitted emitted = EmitModule(mod, &kernel);
  return ffi::String(emitted.mlir);
}

// The SOURCE-ONLY MODULE entry. `DeviceCodegen.build_without_compile` must
// return an IRModule, and both `lower()` paths read `inspect_source()` from it,
// so the `ffi::String` entry above cannot serve as that callback: binding it
// would give callers a str where a module is required.
//
// The module carries the SAME generated Late MLIR as its source, in the format
// `ascModule` already uses for AscendC source modules. It does NOT read the
// frozen P3 binary: a source-only build must work on a host with no device and
// no compiler, which is exactly what the semantics tests need.
ffi::Module BuildHivmSourceOnlyModule(IRModule mod, Target target) {
  CheckHivmTarget(target);
  tirx::PrimFunc kernel;
  Emitted emitted = EmitModule(mod, &kernel);
  return CSourceModuleCreate(emitted.mlir, ffi::String(kSourceOnlyFormat),
                             ffi::Array<ffi::String>());
}

ffi::Module BuildTileLangHivm(IRModule mod, Target target) {
  CheckHivmTarget(target);

  tirx::PrimFunc kernel;
  Emitted emitted = EmitModule(mod, &kernel);

  // The compiler hook is REQUIRED here. Without it there is no binary, and
  // returning a module anyway would be the PREBUILT substitution this entry
  // exists to remove.
  const auto compile = ffi::Function::GetGlobal(kCompileHook);
  TVM_FFI_ICHECK(compile.has_value())
      << "HIVM CODEGEN: no `" << kCompileHook
      << "` is registered, so the generated Late MLIR cannot be compiled. "
         "This entry never falls back to the frozen PREBUILT binary.";
  ffi::Bytes binary =
      (*compile)(ffi::String(emitted.mlir), target).cast<ffi::Bytes>();
  TVM_FFI_ICHECK(binary.size() > 0)
      << "HIVM CODEGEN: the compiler hook returned an empty binary";

  auto fmap = ExtractGeneratedFuncInfo(mod, emitted.symbol);

  // Provenance: this module was generated THIS round, from the MLIR emitted
  // just above. The label names the generator, pins the identity of the input
  // MLIR, and carries the whole MLIR as its trailer, so the GENERATED artefact
  // is traceable to the exact text it came from. It carries no frozen P3 path
  // or digest -- that separation is what the distinct format token exists for.
  const std::string label = GeneratedProvenance(emitted.mlir, emitted.symbol);
  ffi::Map<ffi::String, ffi::String> source_map;
  source_map.Set(ffi::String(ascend::hivm_prebuilt::kHivmGeneratedFormat),
                 ffi::String(label));
  source_map.Set("asc", ffi::String(label));

  return ascend::AscendModuleCreate(
      std::move(binary),
      ffi::String(ascend::hivm_prebuilt::kHivmGeneratedFormat), std::move(fmap),
      std::move(source_map));
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_hivm", BuildTileLangHivm)
      .def(kSourceOnlyHook, BuildHivmSourceOnly)
      .def(kSourceOnlyModuleHook, BuildHivmSourceOnlyModule);
}

} // namespace codegen
} // namespace tvm
