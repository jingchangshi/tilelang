/*!
 * \file hivm_codegen.h
 * \brief Declaration of the real HIVM TIR -> Late MLIR text generator.
 *
 * The generator is a SOURCE-ONLY step: it consumes a prepared TIR PrimFunc and
 * returns MLIR text. It does not run the AscendNPU-IR pipeline, load a binary
 * or touch a device. `rt_mod_ascend_hivm.cc` is the only caller that compiles
 * the returned text, and it is a separate entry so the two can be tested apart.
 */

#ifndef TVM_TL_ASCEND_CODEGEN_HIVM_CODEGEN_H_
#define TVM_TL_ASCEND_CODEGEN_HIVM_CODEGEN_H_

#include <tvm/tirx/function.h>

#include <string>

namespace tvm {
namespace codegen {
namespace hivm {

// Emit Late HIVM/AVE MLIR text for `func`.
//
// Throws `Unsupported` (a std::runtime_error carrying the node type and the
// enclosing function) when the function uses anything outside the supported
// subset. There is no partial output and no fallback: a refusal means no
// module.
std::string EmitLateMlir(const tirx::PrimFunc &func);

// Emit for an explicitly resolved kernel entry symbol. Callers that also build
// the runtime metadata MUST use this overload with the same symbol, so the
// emitted `func.func` name and the FunctionInfo key cannot diverge.
std::string EmitLateMlir(const tirx::PrimFunc &func, const std::string &symbol);

// The ONE kernel entry symbol: the function's `global_symbol` when it has a
// non-empty one, otherwise the name of the GlobalVar that binds it. Throws when
// neither exists, because there is then nothing the runtime could look up.
std::string ResolveKernelSymbol(const tirx::PrimFunc &func,
                                const std::string &global_var_name);

} // namespace hivm
} // namespace codegen
} // namespace tvm

#endif // TVM_TL_ASCEND_CODEGEN_HIVM_CODEGEN_H_
