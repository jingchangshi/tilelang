/*!
 * \file test_ascend_cann_abi_mirror.cc
 * \brief CR-LD-4: bind TileLang's production ACL mirror to the REAL CANN header.
 *
 * WHAT THIS PROVES
 * The Ascend runtime reaches the ACL ABI through hand-written structures, because
 * TileLang keeps no CANN build-time dependency (src/ascend/stubs/ascendcl.h:15-17)
 * and therefore does not compile against the real <acl/acl_rt.h>. A hand-written
 * struct that is merely *plausible* is not safe: a mirror padded to 24 bytes where
 * the real type is 20 compiles cleanly and then mis-loads at runtime. So every
 * layout quantity that crosses the ABI is asserted here against the real header.
 *
 * THE TYPES UNDER TEST ARE THE PRODUCTION TYPES.
 * This TU includes ascend/runtime/acl_rt_types.h -- the same header
 * src/ascend/runtime/ascend_module.cc includes and whose structs the launch path
 * actually fills in. It deliberately does NOT define a second mirror: comparing
 * an independently duplicated test copy against the real header would pass while
 * production stayed wrong.
 *
 * WHY THE STUB IS INCLUDED WITH RENAMED WRAPPERS
 * The shipped stub declares two ACL entrypoints with opaque pointer parameters
 * while the real header declares them with typed parameters. A TU that includes
 * both fails on conflicting extern "C" declarations -- measured, before any
 * assertion runs -- so "put the test in a separate TU" is necessary but not
 * sufficient. The isolation below (include the real header first, macro-rename the
 * two conflicting wrapper identifiers only while including the stub, then #undef)
 * is the technique the Challenger demonstrated. Including the stub here also
 * guards the second half of the claim: the shipped stub really does compile with
 * no CANN types in its own declarations.
 *
 * BUILD (inside the container, via scripts/in-container.sh):
 *   g++ -std=c++17 -Wall -Wextra -Werror \
 *       -I/data/pri/cann-9.2.0/cann/include \
 *       -I/home/s00653124/workspace/tilelang/src \
 *       testing/cpp/test_ascend_cann_abi_mirror.cc -o /tmp/abi_mirror && /tmp/abi_mirror
 */

// 1. The REAL CANN header first, so its declarations own the plain names.
#include <acl/acl_rt.h>

// 2. The shipped stub, with its two conflicting wrapper identifiers renamed for
//    the duration of the include only. The struct-member spellings end in '_' and
//    are distinct tokens, so they are unaffected.
#define aclrtBinaryLoadFromData aclrtBinaryLoadFromData_stub_only
#define aclrtLaunchKernelWithHostArgs aclrtLaunchKernelWithHostArgs_stub_only
#include "ascend/stubs/ascendcl.h"
#undef aclrtBinaryLoadFromData
#undef aclrtLaunchKernelWithHostArgs

// 3. The PRODUCTION mirror -- the types the loader actually uses.
#include "ascend/runtime/acl_rt_types.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace mirror = tvm::tl::ascendcl;

// ---------------------------------------------------------------------------
// Compile-time binding. Each assertion fails the BUILD if the mirror drifts.
// ---------------------------------------------------------------------------

// --- aclrtBinaryLoadOptionType ---------------------------------------------
static_assert(sizeof(mirror::AclBinaryLoadOptionType) ==
                  sizeof(aclrtBinaryLoadOptionType),
              "AclBinaryLoadOptionType size");
static_assert(alignof(mirror::AclBinaryLoadOptionType) ==
                  alignof(aclrtBinaryLoadOptionType),
              "AclBinaryLoadOptionType alignment");
static_assert(
    std::is_same<std::underlying_type<mirror::AclBinaryLoadOptionType>::type,
                 std::underlying_type<aclrtBinaryLoadOptionType>::type>::value,
    "AclBinaryLoadOptionType underlying type");

// --- aclrtBinaryLoadOptionValue --------------------------------------------
static_assert(sizeof(mirror::AclBinaryLoadOptionValue) ==
                  sizeof(aclrtBinaryLoadOptionValue),
              "AclBinaryLoadOptionValue size");
static_assert(alignof(mirror::AclBinaryLoadOptionValue) ==
                  alignof(aclrtBinaryLoadOptionValue),
              "AclBinaryLoadOptionValue alignment");
// Member TYPES, not just offsets: rsv must be the same array type and length.
static_assert(std::is_same<decltype(mirror::AclBinaryLoadOptionValue::isLazyLoad),
                           decltype(aclrtBinaryLoadOptionValue::isLazyLoad)>::value,
              "isLazyLoad type");
static_assert(std::is_same<decltype(mirror::AclBinaryLoadOptionValue::magic),
                           decltype(aclrtBinaryLoadOptionValue::magic)>::value,
              "magic type");
static_assert(
    std::is_same<decltype(mirror::AclBinaryLoadOptionValue::cpuKernelMode),
                 decltype(aclrtBinaryLoadOptionValue::cpuKernelMode)>::value,
    "cpuKernelMode type");
static_assert(std::is_same<decltype(mirror::AclBinaryLoadOptionValue::rsv),
                           decltype(aclrtBinaryLoadOptionValue::rsv)>::value,
              "rsv type and reserved-array length");
// offsetof of EVERY member, including the union's. Every union member shares
// offset 0; asserting it keeps the coverage uniform across both unions rather
// than checking one and assuming the other.
static_assert(offsetof(mirror::AclBinaryLoadOptionValue, isLazyLoad) == 0,
              "AclBinaryLoadOptionValue::isLazyLoad offset");
static_assert(offsetof(mirror::AclBinaryLoadOptionValue, magic) == 0,
              "AclBinaryLoadOptionValue::magic offset");
static_assert(offsetof(mirror::AclBinaryLoadOptionValue, cpuKernelMode) == 0,
              "AclBinaryLoadOptionValue::cpuKernelMode offset");
static_assert(offsetof(mirror::AclBinaryLoadOptionValue, rsv) == 0,
              "AclBinaryLoadOptionValue::rsv offset");

// --- aclrtBinaryLoadOption -------------------------------------------------
static_assert(sizeof(mirror::AclBinaryLoadOption) == sizeof(aclrtBinaryLoadOption),
              "AclBinaryLoadOption size");
static_assert(alignof(mirror::AclBinaryLoadOption) ==
                  alignof(aclrtBinaryLoadOption),
              "AclBinaryLoadOption alignment");
static_assert(offsetof(mirror::AclBinaryLoadOption, type) ==
                  offsetof(aclrtBinaryLoadOption, type),
              "AclBinaryLoadOption::type offset");
static_assert(offsetof(mirror::AclBinaryLoadOption, value) ==
                  offsetof(aclrtBinaryLoadOption, value),
              "AclBinaryLoadOption::value offset");
// `type` is an enum on both sides; bind it by width so a mirror that used a
// plain int (same width, different type category) is still caught by the
// underlying-type check above, and a mirror that used a narrower enum is caught
// here.
static_assert(sizeof(mirror::AclBinaryLoadOption::type) ==
                  sizeof(aclrtBinaryLoadOption::type),
              "AclBinaryLoadOption::type width");

// --- aclrtBinaryLoadOptions ------------------------------------------------
static_assert(sizeof(mirror::AclBinaryLoadOptions) ==
                  sizeof(aclrtBinaryLoadOptions),
              "AclBinaryLoadOptions size");
static_assert(alignof(mirror::AclBinaryLoadOptions) ==
                  alignof(aclrtBinaryLoadOptions),
              "AclBinaryLoadOptions alignment");
static_assert(offsetof(mirror::AclBinaryLoadOptions, options) ==
                  offsetof(aclrtBinaryLoadOptions, options),
              "AclBinaryLoadOptions::options offset");
static_assert(offsetof(mirror::AclBinaryLoadOptions, numOpt) ==
                  offsetof(aclrtBinaryLoadOptions, numOpt),
              "AclBinaryLoadOptions::numOpt offset");
// Member TYPES as well as offsets. A same-size, same-offset field swap
// (size_t -> uint32_t) is invisible to sizeof/alignof/offsetof alone, so the
// width is asserted for every member. Pointer and enum members are compared by
// kind and width rather than by is_same, because each side names its own type.
static_assert(std::is_same<decltype(mirror::AclBinaryLoadOptions::numOpt),
                           decltype(aclrtBinaryLoadOptions::numOpt)>::value,
              "AclBinaryLoadOptions::numOpt type");
static_assert(
    std::is_pointer<decltype(mirror::AclBinaryLoadOptions::options)>::value &&
        std::is_pointer<decltype(aclrtBinaryLoadOptions::options)>::value,
    "AclBinaryLoadOptions::options is a pointer");
static_assert(sizeof(mirror::AclBinaryLoadOptions::options) ==
                  sizeof(aclrtBinaryLoadOptions::options),
              "AclBinaryLoadOptions::options pointer width");

// --- aclrtEngineType -------------------------------------------------------
static_assert(sizeof(mirror::AclEngineType) == sizeof(aclrtEngineType),
              "AclEngineType size");
static_assert(alignof(mirror::AclEngineType) == alignof(aclrtEngineType),
              "AclEngineType alignment");

// --- aclrtTimeoutUs --------------------------------------------------------
static_assert(sizeof(mirror::AclTimeoutUs) == sizeof(aclrtTimeoutUs),
              "AclTimeoutUs size");
static_assert(alignof(mirror::AclTimeoutUs) == alignof(aclrtTimeoutUs),
              "AclTimeoutUs alignment");
static_assert(offsetof(mirror::AclTimeoutUs, timeoutLow) ==
                  offsetof(aclrtTimeoutUs, timeoutLow),
              "AclTimeoutUs::timeoutLow offset");
static_assert(offsetof(mirror::AclTimeoutUs, timeoutHigh) ==
                  offsetof(aclrtTimeoutUs, timeoutHigh),
              "AclTimeoutUs::timeoutHigh offset");

// --- aclrtLaunchKernelAttrId ----------------------------------------------
static_assert(sizeof(mirror::AclLaunchKernelAttrId) ==
                  sizeof(aclrtLaunchKernelAttrId),
              "AclLaunchKernelAttrId size");
static_assert(alignof(mirror::AclLaunchKernelAttrId) ==
                  alignof(aclrtLaunchKernelAttrId),
              "AclLaunchKernelAttrId alignment");

// --- aclrtLaunchKernelAttrValue -------------------------------------------
static_assert(sizeof(mirror::AclLaunchKernelAttrValue) ==
                  sizeof(aclrtLaunchKernelAttrValue),
              "AclLaunchKernelAttrValue size");
static_assert(alignof(mirror::AclLaunchKernelAttrValue) ==
                  alignof(aclrtLaunchKernelAttrValue),
              "AclLaunchKernelAttrValue alignment");
static_assert(
    std::is_same<decltype(mirror::AclLaunchKernelAttrValue::schemMode),
                 decltype(aclrtLaunchKernelAttrValue::schemMode)>::value,
    "schemMode type");
// localMemorySize is marked ACL_DEPRECATED_MESSAGE in the real header, but
// CR-LD-4 requires the union's member types to be bound -- and a deprecated
// member still occupies ABI space. Suppress the diagnostic for this one access
// only; the deprecation is a usage advisory, not a layout change.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
static_assert(
    std::is_same<decltype(mirror::AclLaunchKernelAttrValue::localMemorySize),
                 decltype(aclrtLaunchKernelAttrValue::localMemorySize)>::value,
    "localMemorySize type");
#pragma GCC diagnostic pop
static_assert(
    std::is_same<decltype(mirror::AclLaunchKernelAttrValue::dynUBufSize),
                 decltype(aclrtLaunchKernelAttrValue::dynUBufSize)>::value,
    "dynUBufSize type");
static_assert(
    std::is_same<decltype(mirror::AclLaunchKernelAttrValue::blockDimOffset),
                 decltype(aclrtLaunchKernelAttrValue::blockDimOffset)>::value,
    "blockDimOffset type");
static_assert(
    std::is_same<decltype(mirror::AclLaunchKernelAttrValue::isBlockTaskPrefetch),
                 decltype(aclrtLaunchKernelAttrValue::isBlockTaskPrefetch)>::value,
    "isBlockTaskPrefetch type");
static_assert(
    std::is_same<decltype(mirror::AclLaunchKernelAttrValue::isDataDump),
                 decltype(aclrtLaunchKernelAttrValue::isDataDump)>::value,
    "isDataDump type");
static_assert(
    std::is_same<decltype(mirror::AclLaunchKernelAttrValue::timeout),
                 decltype(aclrtLaunchKernelAttrValue::timeout)>::value,
    "timeout type");
static_assert(
    std::is_same<decltype(mirror::AclLaunchKernelAttrValue::rsv),
                 decltype(aclrtLaunchKernelAttrValue::rsv)>::value,
    "rsv type and reserved-array length");
// engineType is a named enum in both; compare its layout, not its spelling.
static_assert(
    sizeof(mirror::AclLaunchKernelAttrValue::engineType) ==
        sizeof(aclrtLaunchKernelAttrValue::engineType),
    "engineType size");
// timeoutUs is a nested struct in both.
static_assert(
    std::is_same<decltype(mirror::AclLaunchKernelAttrValue::timeoutUs),
                 decltype(aclrtLaunchKernelAttrValue::timeoutUs)>::value ==
        false,
    "timeoutUs is a distinct named struct type in each namespace");
static_assert(
    sizeof(mirror::AclLaunchKernelAttrValue::timeoutUs) ==
        sizeof(aclrtLaunchKernelAttrValue::timeoutUs),
    "timeoutUs size");
// Every union member shares offset 0; assert it explicitly anyway.
static_assert(offsetof(mirror::AclLaunchKernelAttrValue, schemMode) == 0,
              "AclLaunchKernelAttrValue::schemMode offset");
static_assert(offsetof(mirror::AclLaunchKernelAttrValue, localMemorySize) == 0,
              "AclLaunchKernelAttrValue::localMemorySize offset");
static_assert(offsetof(mirror::AclLaunchKernelAttrValue, dynUBufSize) == 0,
              "AclLaunchKernelAttrValue::dynUBufSize offset");
static_assert(offsetof(mirror::AclLaunchKernelAttrValue, engineType) == 0,
              "AclLaunchKernelAttrValue::engineType offset");
static_assert(offsetof(mirror::AclLaunchKernelAttrValue, blockDimOffset) == 0,
              "AclLaunchKernelAttrValue::blockDimOffset offset");
static_assert(offsetof(mirror::AclLaunchKernelAttrValue, isBlockTaskPrefetch) == 0,
              "AclLaunchKernelAttrValue::isBlockTaskPrefetch offset");
static_assert(offsetof(mirror::AclLaunchKernelAttrValue, isDataDump) == 0,
              "AclLaunchKernelAttrValue::isDataDump offset");
static_assert(offsetof(mirror::AclLaunchKernelAttrValue, timeout) == 0,
              "AclLaunchKernelAttrValue::timeout offset");
static_assert(offsetof(mirror::AclLaunchKernelAttrValue, timeoutUs) == 0,
              "AclLaunchKernelAttrValue::timeoutUs offset");
static_assert(offsetof(mirror::AclLaunchKernelAttrValue, rsv) == 0,
              "AclLaunchKernelAttrValue::rsv offset");

// --- aclrtLaunchKernelAttr ------------------------------------------------
static_assert(sizeof(mirror::AclLaunchKernelAttr) == sizeof(aclrtLaunchKernelAttr),
              "AclLaunchKernelAttr size");
static_assert(alignof(mirror::AclLaunchKernelAttr) ==
                  alignof(aclrtLaunchKernelAttr),
              "AclLaunchKernelAttr alignment");
static_assert(offsetof(mirror::AclLaunchKernelAttr, id) ==
                  offsetof(aclrtLaunchKernelAttr, id),
              "AclLaunchKernelAttr::id offset");
static_assert(offsetof(mirror::AclLaunchKernelAttr, value) ==
                  offsetof(aclrtLaunchKernelAttr, value),
              "AclLaunchKernelAttr::value offset");

// --- aclrtLaunchKernelCfg -------------------------------------------------
static_assert(sizeof(mirror::AclLaunchKernelCfg) == sizeof(aclrtLaunchKernelCfg),
              "AclLaunchKernelCfg size");
static_assert(alignof(mirror::AclLaunchKernelCfg) ==
                  alignof(aclrtLaunchKernelCfg),
              "AclLaunchKernelCfg alignment");
static_assert(offsetof(mirror::AclLaunchKernelCfg, attrs) ==
                  offsetof(aclrtLaunchKernelCfg, attrs),
              "AclLaunchKernelCfg::attrs offset");
static_assert(offsetof(mirror::AclLaunchKernelCfg, numAttrs) ==
                  offsetof(aclrtLaunchKernelCfg, numAttrs),
              "AclLaunchKernelCfg::numAttrs offset");
static_assert(std::is_same<decltype(mirror::AclLaunchKernelCfg::numAttrs),
                           decltype(aclrtLaunchKernelCfg::numAttrs)>::value,
              "AclLaunchKernelCfg::numAttrs type");
static_assert(
    std::is_pointer<decltype(mirror::AclLaunchKernelCfg::attrs)>::value &&
        std::is_pointer<decltype(aclrtLaunchKernelCfg::attrs)>::value,
    "AclLaunchKernelCfg::attrs is a pointer");
static_assert(sizeof(mirror::AclLaunchKernelCfg::attrs) ==
                  sizeof(aclrtLaunchKernelCfg::attrs),
              "AclLaunchKernelCfg::attrs pointer width");

// ---------------------------------------------------------------------------
// Mirrored enumerators and magic constants, bound at COMPILE time.
// A run-time comparison alone is too weak: a mirror whose constant is wrong
// still compiles, so a compile-only check reports success for a wrong mirror.
// These static_asserts make the build itself fail, and the run-time report in
// main() additionally prints which value drifted.
// ---------------------------------------------------------------------------
// The two sides are DISTINCT enum types (that is the point of the Acl* prefix),
// so a direct == would be an enum-compare diagnostic rather than a value check.
// Each operand is converted to its underlying integer type first, which is what
// makes the comparison about the VALUE and not about the type identity -- an
// identity already asserted separately above.
template <typename T>
constexpr long long AclValueOf(T v) {
  return static_cast<long long>(static_cast<typename std::underlying_type<T>::type>(v));
}

static_assert(AclValueOf(mirror::kAclBinaryLoadOptLazyLoad) == ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD, "AclBinaryLoadOptLazyLoad value");
static_assert(AclValueOf(mirror::kAclBinaryLoadOptLazyMagic) == ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC, "AclBinaryLoadOptLazyMagic value");
static_assert(AclValueOf(mirror::kAclBinaryLoadOptMagic) == ACL_RT_BINARY_LOAD_OPT_MAGIC, "AclBinaryLoadOptMagic value");
static_assert(AclValueOf(mirror::kAclBinaryLoadOptCpuKernelMode) == ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, "AclBinaryLoadOptCpuKernelMode value");
static_assert(AclValueOf(mirror::kAclEngineTypeAic) == ACL_RT_ENGINE_TYPE_AIC, "AclEngineTypeAic value");
static_assert(AclValueOf(mirror::kAclEngineTypeAiv) == ACL_RT_ENGINE_TYPE_AIV, "AclEngineTypeAiv value");
static_assert(AclValueOf(mirror::kAclLaunchKernelAttrSchemMode) == ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE, "AclLaunchKernelAttrSchemMode value");
static_assert(AclValueOf(mirror::kAclLaunchKernelAttrDynUbufSize) == ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE, "AclLaunchKernelAttrDynUbufSize value");
static_assert(AclValueOf(mirror::kAclLaunchKernelAttrEngineType) == ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE, "AclLaunchKernelAttrEngineType value");
static_assert(AclValueOf(mirror::kAclLaunchKernelAttrBlockdimOffset) == ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET, "AclLaunchKernelAttrBlockdimOffset value");
static_assert(AclValueOf(mirror::kAclLaunchKernelAttrBlockTaskPrefetch) == ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH, "AclLaunchKernelAttrBlockTaskPrefetch value");
static_assert(AclValueOf(mirror::kAclLaunchKernelAttrDataDump) == ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP, "AclLaunchKernelAttrDataDump value");
static_assert(AclValueOf(mirror::kAclLaunchKernelAttrTimeout) == ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT, "AclLaunchKernelAttrTimeout value");
static_assert(AclValueOf(mirror::kAclLaunchKernelAttrTimeoutUs) == ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US, "AclLaunchKernelAttrTimeoutUs value");
// The deprecated enumerator is named under the same narrow suppression.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
static_assert(AclValueOf(mirror::kAclLaunchKernelAttrLocalMemorySize) ==
                  ACL_RT_LAUNCH_KERNEL_ATTR_LOCAL_MEMORY_SIZE,
              "AclLaunchKernelAttrLocalMemorySize value");
#pragma GCC diagnostic pop
// Magic constants.
static_assert(mirror::kAclBinaryMagicElfAicore == ACL_RT_BINARY_MAGIC_ELF_AICORE,
              "kAclBinaryMagicElfAicore value");
static_assert(mirror::kAclBinaryMagicElfVectorCore == ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE,
              "kAclBinaryMagicElfVectorCore value");
static_assert(mirror::kAclBinaryMagicElfCubeCore == ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE,
              "kAclBinaryMagicElfCubeCore value");

// ---------------------------------------------------------------------------
// Runtime report: the same values, printed so a drift names itself.
// ---------------------------------------------------------------------------
namespace {

int g_failures = 0;

void ExpectEq(const char *what, long long mirrored, long long real) {
  const bool ok = (mirrored == real);
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-46s mirror=%-12lld real=%-12lld %s\n", what, mirrored, real,
              ok ? "OK" : "MISMATCH");
}

template <typename T, typename U>
void ExpectSameSize(const char *what) {
  const long long a = static_cast<long long>(sizeof(T));
  const long long b = static_cast<long long>(sizeof(U));
  const bool ok = (a == b);
  if (!ok) {
    ++g_failures;
  }
  std::printf("%-46s mirror=%-12lld real=%-12lld %s\n", what, a, b,
              ok ? "OK" : "MISMATCH");
}

} // namespace

int main() {
  std::printf("=== mirrored enumerators vs real header ===\n");
  ExpectEq("ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD",
           mirror::kAclBinaryLoadOptLazyLoad, ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD);
  ExpectEq("ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC",
           mirror::kAclBinaryLoadOptLazyMagic, ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC);
  ExpectEq("ACL_RT_BINARY_LOAD_OPT_MAGIC", mirror::kAclBinaryLoadOptMagic,
           ACL_RT_BINARY_LOAD_OPT_MAGIC);
  ExpectEq("ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE",
           mirror::kAclBinaryLoadOptCpuKernelMode,
           ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE);
  ExpectEq("ACL_RT_ENGINE_TYPE_AIC", mirror::kAclEngineTypeAic,
           ACL_RT_ENGINE_TYPE_AIC);
  ExpectEq("ACL_RT_ENGINE_TYPE_AIV", mirror::kAclEngineTypeAiv,
           ACL_RT_ENGINE_TYPE_AIV);
  ExpectEq("ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE",
           mirror::kAclLaunchKernelAttrSchemMode,
           ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE);
// The deprecated LOCAL_MEMORY_SIZE enumerator shares value 2 with
// DYN_UBUF_SIZE; it is asserted under the same narrow suppression because
// naming it is what the real header warns about.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  ExpectEq("ACL_RT_LAUNCH_KERNEL_ATTR_LOCAL_MEMORY_SIZE",
           mirror::kAclLaunchKernelAttrLocalMemorySize,
           ACL_RT_LAUNCH_KERNEL_ATTR_LOCAL_MEMORY_SIZE);
#pragma GCC diagnostic pop
  ExpectEq("ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE",
           mirror::kAclLaunchKernelAttrDynUbufSize,
           ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE);
  ExpectEq("ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE",
           mirror::kAclLaunchKernelAttrEngineType,
           ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE);
  ExpectEq("ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET",
           mirror::kAclLaunchKernelAttrBlockdimOffset,
           ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET);
  ExpectEq("ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH",
           mirror::kAclLaunchKernelAttrBlockTaskPrefetch,
           ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH);
  ExpectEq("ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP",
           mirror::kAclLaunchKernelAttrDataDump,
           ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP);
  ExpectEq("ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT",
           mirror::kAclLaunchKernelAttrTimeout,
           ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT);
  ExpectEq("ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US",
           mirror::kAclLaunchKernelAttrTimeoutUs,
           ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US);

  std::printf("=== mirrored magic constants vs real header ===\n");
  ExpectEq("ACL_RT_BINARY_MAGIC_ELF_AICORE", mirror::kAclBinaryMagicElfAicore,
           ACL_RT_BINARY_MAGIC_ELF_AICORE);
  ExpectEq("ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE",
           mirror::kAclBinaryMagicElfVectorCore,
           ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE);
  ExpectEq("ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE",
           mirror::kAclBinaryMagicElfCubeCore,
           ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE);

  std::printf("=== layout report (mirror vs real) ===\n");
  ExpectSameSize<mirror::AclBinaryLoadOptions, aclrtBinaryLoadOptions>(
      "sizeof(AclBinaryLoadOptions)");
  ExpectSameSize<mirror::AclBinaryLoadOption, aclrtBinaryLoadOption>(
      "sizeof(AclBinaryLoadOption)");
  ExpectSameSize<mirror::AclBinaryLoadOptionValue, aclrtBinaryLoadOptionValue>(
      "sizeof(AclBinaryLoadOptionValue)");
  ExpectSameSize<mirror::AclLaunchKernelAttr, aclrtLaunchKernelAttr>(
      "sizeof(AclLaunchKernelAttr)");
  ExpectSameSize<mirror::AclLaunchKernelAttrValue, aclrtLaunchKernelAttrValue>(
      "sizeof(AclLaunchKernelAttrValue)");
  ExpectSameSize<mirror::AclLaunchKernelCfg, aclrtLaunchKernelCfg>(
      "sizeof(AclLaunchKernelCfg)");
  std::printf("%-46s mirror=%-12zu real=%-12zu %s\n",
              "offsetof(AclBinaryLoadOption, value)",
              offsetof(mirror::AclBinaryLoadOption, value),
              offsetof(aclrtBinaryLoadOption, value),
              offsetof(mirror::AclBinaryLoadOption, value) ==
                      offsetof(aclrtBinaryLoadOption, value)
                  ? "OK"
                  : "MISMATCH");
  std::printf("%-46s mirror=%-12zu real=%-12zu %s\n",
              "offsetof(AclLaunchKernelAttr, value)",
              offsetof(mirror::AclLaunchKernelAttr, value),
              offsetof(aclrtLaunchKernelAttr, value),
              offsetof(mirror::AclLaunchKernelAttr, value) ==
                      offsetof(aclrtLaunchKernelAttr, value)
                  ? "OK"
                  : "MISMATCH");

  std::printf("=== the shipped stub compiled alongside the real header ===\n");
  std::printf("%-46s %s\n", "ascend/stubs/ascendcl.h included",
              "OK (renamed wrappers, no conflict)");

  std::printf("\nRESULT: %s (%d mismatches)\n",
              g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
