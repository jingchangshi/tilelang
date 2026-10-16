#include "codegen_ascend.h"

#include "ascend/hivm_prebuilt_identity.h"
#include "ascend/codegen/hivm_prebuilt_contract.h"
#include "ascend/codegen/sha256.h"
#include "runtime/metadata.h"
#include "support/check.h"
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

#include <fstream>
#include <string>
#include <vector>

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

std::string AscendCodeGen(IRModule mod) {
  bool output_ssa = false;
  CodeGenTileLangAscend cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangAscend: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    auto is_global_func =
        f->GetAttr<Bool>(tvm::tirx::attr::kIsGlobalFunc).value_or(Bool(false));
    // For Ascend, we accept functions that are either:
    // 1. Device kernel launches (calling_conv == kDeviceKernelLaunch), or
    // 2. Global functions called via call_extern (is_global_func == true)
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch || is_global_func)
        << "CodeGenTileLangAscend: Function must have "
           "calling_conv=kDeviceKernelLaunch "
        << "or be marked as a global function (tir.is_global_func=True)";
    cg.AddFunction(f);
  }

  return cg.Finish();
}

// Build the per-kernel FunctionInfo table the runtime module needs to recover
// the launch grid (via LaunchParamConfig) and pack arguments. Mirrors the CUDA
// ExtractFuncInfo (src/cuda/codegen/rt_mod_cuda.cc) but only carries what the
// Ascend launch ABI uses: arg types + kernel launch params (thread extents).
ffi::Map<ffi::String, runtime::FunctionInfo>
ExtractFuncInfo(const IRModule &mod) {
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap;
  for (auto kv : mod->functions) {
    auto f = Downcast<tirx::PrimFunc>(kv.second);
    // Only device kernels are launchable; skip call_extern globals.
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    if (!(calling_conv == CallingConv::kDeviceKernelLaunch)) {
      continue;
    }

    ffi::Array<DLDataType> arg_types;
    ffi::Array<ffi::String> launch_param_tags;

    for (size_t i = 0; i < f->params.size(); ++i) {
      DataType dtype = f->params[i].dtype();
      if (dtype.is_bool())
        dtype = DataType::Int(32);
      arg_types.push_back(dtype);
    }
    if (auto opt = f->GetAttr<ffi::Array<ffi::String>>(
            tirx::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        launch_param_tags.push_back(tag);
      }
    }

    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    std::string sym = global_symbol.has_value()
                          ? static_cast<std::string>(global_symbol.value())
                          : static_cast<std::string>(
                                Downcast<GlobalVar>(kv.first)->name_hint);
    fmap.Set(ffi::String(sym),
             runtime::FunctionInfo(ffi::String(sym), arg_types,
                                   launch_param_tags, {}));
  }
  return fmap;
}

} // namespace

ffi::Module BuildTileLangAscend(IRModule mod, Target target) {
  std::string code = AscendCodeGen(mod);

  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_ascend_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  std::string aibin;
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_ascend_compile")) {
    tvm::transform::PassContext pass_ctx =
        tvm::transform::PassContext::Current();
    aibin = (*f)(code, target, pass_ctx->config).cast<std::string>();
  } else {
    ICHECK(false) << "tilelang_callback_ascend_compile is not registered";
  }

  ffi::Map<ffi::String, ffi::String> source_map;
  source_map.Set("asc", ffi::String(code));
  return ascend::AscendModuleCreate(ffi::Bytes(aibin.data(), aibin.size()),
                                    ffi::String("aibin"), ExtractFuncInfo(mod),
                                    std::move(source_map));
}

ffi::Module BuildTileLangAscendWithoutCompile(IRModule mod, Target target) {
  std::string code = AscendCodeGen(mod);
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_ascend_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  // Source-only module used by the cython adapter / inspect_source. The cython
  // path compiles host+device together with bisheng, so no runnable module is
  // needed here.
  return CSourceModuleCreate(code, "asc", ffi::Array<ffi::String>());
}

// ===========================================================================
// TEST_ONLY PREBUILT HIVM entry (goal section 5.1)
// ===========================================================================
// This entry does NOT run AscendC source codegen, does NOT run PTO codegen and
// does NOT run any HIVM TIR->Late MLIR step. It injects the single frozen P3
// device binary into the EXISTING Ascend runtime module, so a normal TileLang
// lowering -> host codegen -> TVM-FFI launch carries that binary to the device.
//
// It is deliberately NOT a DeviceCodeGen: nothing here turns TIR into device
// code. Reporting a module built by this entry as DeviceCodeGen E2E would be
// false (acceptance CR-PB-2/CR-X-8).

// Read the whole file ONCE. CR-ID-2 requires that the bytes hashed are the
// bytes loaded: re-opening the path between the identity check and the module
// creation would let the two differ, so the caller below hashes this buffer and
// hands THIS buffer to AscendModuleCreate.
std::string ReadFrozenBinary(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  TVM_FFI_ICHECK(input.good())
      << "HIVM PREBUILT identity: cannot open the frozen device binary at '"
      << path << "'";
  std::string bytes((std::istreambuf_iterator<char>(input)),
                    std::istreambuf_iterator<char>());
  TVM_FFI_ICHECK(!input.bad())
      << "HIVM PREBUILT identity: read error on the frozen device binary at '"
      << path << "'";
  return bytes;
}

// The path the entry reads. It is the frozen pin by default; a TEST_ONLY
// pass-config override exists so the negative controls (CR-REJ-1) can point at
// a CORRUPTED COPY without the frozen file ever being modified. The override
// changes only WHERE the bytes come from -- the sha256 and size pins below are
// unconditional, so an override can only ever produce a REFUSAL, never a
// different accepted binary.
//
// The option is REGISTERED (TVM_REGISTER_PASS_CONFIG_OPTION) rather than read
// as an unknown key: PassContext rejects unregistered keys outright, so an
// unregistered read would raise AttributeError and the negatives would look
// "refused" for the wrong reason -- a check that always fires reads as
// discrimination.
constexpr const char *kHivmPrebuiltBinaryPathConfig =
    "tl.hivm_prebuilt_binary_path";

std::string HivmPrebuiltBinaryPath() {
  namespace ident = tvm::ascend::hivm_prebuilt;
  auto ctx = tvm::transform::PassContext::Current();
  auto override_path = ctx->GetConfig<ffi::String>(
      kHivmPrebuiltBinaryPathConfig, ffi::Optional<ffi::String>());
  if (override_path.has_value()) {
    return static_cast<std::string>(override_path.value());
  }
  return ident::kFrozenBinaryPath;
}

ffi::Module BuildTileLangHivmPrebuilt(IRModule mod, Target target) {
  namespace ident = tvm::ascend::hivm_prebuilt;

  // (1) The target must be an EXPLICIT HIVM selection. This is the first check
  // so that a plain AscendC/PTO target reaching this entry by mistake is
  // refused before any file is touched.
  TVM_FFI_ICHECK(target.defined())
      << "HIVM PREBUILT: target is undefined";
  TVM_FFI_ICHECK(target->kind->name == "ascend")
      << "HIVM PREBUILT: expected target kind 'ascend', got '"
      << target->kind->name << "'";
  bool has_hivm = false;
  for (const auto &k : target->keys) {
    if (static_cast<std::string>(k) == "hivm") {
      has_hivm = true;
    }
  }
  TVM_FFI_ICHECK(has_hivm)
      << "HIVM PREBUILT: the target must carry the explicit 'hivm' key; this "
         "entry is never reached through plain AscendC or PTO";

  // (2) Binary identity. Any sha256/size mismatch is refused HERE, before a
  // module exists and therefore before any ACL load can reach a device
  // (CR-ID-1/CR-ID-3). The message names the identity check, because CR-REJ-1
  // excludes reading a downstream loader failure as a satisfied check.
  const std::string binary_path = HivmPrebuiltBinaryPath();
  const std::string bytes = ReadFrozenBinary(binary_path);
  const std::string digest = hivm_prebuilt::Sha256Hex(bytes);
  TVM_FFI_ICHECK(bytes.size() == ident::kFrozenBinarySize)
      << "HIVM PREBUILT binary identity check FAILED: size of '"
      << binary_path << "' is " << bytes.size() << " bytes, the "
      << "frozen program is " << ident::kFrozenBinarySize << " bytes";
  TVM_FFI_ICHECK(digest == ident::kFrozenBinarySha256)
      << "HIVM PREBUILT binary identity check FAILED: sha256 of '"
      << binary_path << "' is " << digest << ", the frozen "
      << "program is " << ident::kFrozenBinarySha256;

  // (3) Fixed-program contract, BEFORE the module is created (BC-2). ExtractFuncInfo
  // alone copies whatever the prepared TIR declares and enforces no contract, so
  // a correct binary can be paired with incompatible prepared TIR and produce an
  // internally consistent but WRONG host ABI.
  hivm_prebuilt::CheckFixedProgramContract(mod, target);

  // (4) The EXISTING FunctionInfo extractor, on the prepared device IRModule.
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap = ExtractFuncInfo(mod);

  // (5) The honest TEST_ONLY label, under BOTH the format key and "asc".
  // AscendModuleNode::InspectSource("") falls back only to the "asc" entry, so
  // a source map keyed only by the new format would read back EMPTY on default
  // inspection (BC-4).
  const std::string label = ident::HivmPrebuiltSourceLabel();
  ffi::Map<ffi::String, ffi::String> source_map;
  source_map.Set(ffi::String(ident::kHivmPrebuiltFormat), ffi::String(label));
  source_map.Set("asc", ffi::String(label));

  // (6) The format token carries the loader policy (VECTOR_CORE), so it travels
  // in the serialized payload instead of process-global state (goal 5.4).
  return ascend::AscendModuleCreate(
      ffi::Bytes(bytes.data(), bytes.size()),
      ffi::String(ident::kHivmPrebuiltFormat), std::move(fmap),
      std::move(source_map));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.ascend", BuildTileLangAscend)
      .def("target.build.tilelang_ascend", BuildTileLangAscend)
      .def("target.build.tilelang_ascend_without_compile",
           BuildTileLangAscendWithoutCompile)
      .def("target.build.tilelang_hivm_prebuilt", BuildTileLangHivmPrebuilt);
}

// TEST_ONLY: lets the negative controls point the identity check at a corrupted
// COPY. Registered so PassContext accepts the key; see HivmPrebuiltBinaryPath.
TVM_REGISTER_PASS_CONFIG_OPTION(kHivmPrebuiltBinaryPathConfig, ffi::String);

} // namespace codegen
} // namespace tvm
