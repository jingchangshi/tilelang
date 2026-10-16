/*!
 * \file hivm_prebuilt_identity.h
 * \brief The ONE definition of the PREBUILT HIVM entry's identity.
 *
 * WHY THIS FILE EXISTS
 * Three places must agree on the same facts:
 *   1. the codegen entry (src/ascend/codegen/rt_mod_ascend.cc) checks the frozen
 *      binary's sha256 and size before it creates a module;
 *   2. the runtime module carries the loader-policy format token;
 *   3. serialization restore (ascend_module.cc) must RECONSTRUCT the honest
 *      PREBUILT label from retained module information, because
 *      AscendModuleLoadFromBytes restores an EMPTY source map (BC-4).
 * If each site spelled the path, the digest or the label itself, a restore
 * could silently describe a different binary than the one that was checked --
 * and the label is exactly what CR-PB-1 asks a probe to read back. So the pin,
 * the format token and the label live here once, and both the codegen and the
 * runtime include this header.
 *
 * SCOPE: TEST_ONLY. This header describes the single frozen P3 device program
 * that P4 is permitted to load (goal section 3). It is not a general facility,
 * and nothing here runs AscendC/PTO codegen or any HIVM TIR->Late MLIR step.
 */

#ifndef TVM_TL_ASCEND_HIVM_PREBUILT_IDENTITY_H_
#define TVM_TL_ASCEND_HIVM_PREBUILT_IDENTITY_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace tvm {
namespace ascend {
namespace hivm_prebuilt {

// The frozen P3 device program (goal section 3). Both values were re-measured
// from the file, not transcribed from the goal: see the evidence row in
// hivm-poc-evidence/p4-20261016/impl/p4b-prebuilt/.
inline constexpr const char *kFrozenBinaryPath =
    "/home/s00653124/workspace/hivm-poc-evidence/"
    "p2-20261013-late-pipeline/verify/work/run/A1/kernel.o";
inline constexpr const char *kFrozenBinarySha256 =
    "2c4247c3b2ce11f79ad15415eb06d4ce1d5523a0c19c91ad3743bc08e7b8c4c9";
inline constexpr size_t kFrozenBinarySize = 4448;

// The module format token. It is the loader-policy carrier (work order
// P4-B.2): the format travels in the serialized payload as
// [fmt][fmap][code], so the VECTOR_CORE decision is a property of the module
// rather than of process-global state or an environment variable -- which
// goal section 5.4 forbids. The existing AscendC modules keep "aibin"
// unchanged.
inline constexpr const char *kAibinFormat = "aibin";
inline constexpr const char *kHivmPrebuiltFormat = "hivm-aibin";

// The auditable TEST_ONLY description returned by inspect_source (CR-PB-1).
// It states the PREBUILT origin and names the frozen binary's source path. It
// deliberately contains NO generated-source text of any kind: P4 generated
// neither the AscendC source nor any HIVM late-stage IR (CR-PB-3). The wording
// avoids even naming those artifacts, so a probe that greps the label for them
// cannot be satisfied by a negation sentence -- a label that says "no X was
// generated" still CONTAINS X.
inline std::string HivmPrebuiltSourceLabel() {
  std::string label;
  label += "// TEST_ONLY PREBUILT device binary -- NOT DeviceCodeGen output.\n";
  label += "// This module injects a FROZEN device binary produced by the P3 "
           "late-pipeline round.\n";
  label += "// No device code was generated for it this round: this entry only "
           "loads the\n";
  label += "// frozen bytes below, so the text of this label is not code.\n";
  label += "// PREBUILT source: ";
  label += kFrozenBinaryPath;
  label += "\n// sha256: ";
  label += kFrozenBinarySha256;
  label += "\n// size: " + std::to_string(kFrozenBinarySize) + " bytes\n";
  label += "// kernel symbol: main_kernel; grid=64; dynamic UB=196608\n";
  return label;
}

// The format token that selects the VECTOR_CORE loader policy. Kept as a
// predicate so the loader and the restore path cannot disagree about which
// token means which policy.
inline bool FormatUsesVectorCoreLoader(const std::string &fmt) {
  return fmt == kHivmPrebuiltFormat;
}

} // namespace hivm_prebuilt
} // namespace ascend
} // namespace tvm

#endif // TVM_TL_ASCEND_HIVM_PREBUILT_IDENTITY_H_
