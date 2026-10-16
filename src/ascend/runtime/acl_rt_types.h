/*!
 * \file acl_rt_types.h
 * \brief Types-only mirror of the CANN ACL structures the Ascend runtime uses.
 *
 * WHY THIS FILE EXISTS
 * TileLang deliberately keeps no CANN build-time dependency: the runtime
 * resolves libascendcl.so lazily through the stub in src/ascend/stubs/ (see the
 * comment at src/ascend/stubs/ascendcl.h:15-17). The real <acl/acl_rt.h> is
 * therefore NOT on the production include path, so the structures that cross
 * the ACL ABI must be expressed here.
 *
 * This header is the SINGLE production definition of those structures. It holds
 * TYPES ONLY -- no function declarations and no wrapper prototypes -- so a test
 * translation unit can include it alongside the real CANN header without
 * colliding on the two wrapper signatures the stub deliberately declares with
 * opaque pointer parameters (aclrtBinaryLoadFromData and
 * aclrtLaunchKernelWithHostArgs). That collision is real and was measured: a TU
 * including both the shipped stub and the real header fails on conflicting
 * extern "C" declarations before reaching any assertion.
 *
 * HOW IT IS VALIDATED
 * testing/cpp/test_ascend_cann_abi_mirror.cc compiles these types beside the real
 * <acl/acl_rt.h> and static_asserts every sizeof, alignof, member offsetof, union
 * member size, reserved-array length, enumerator and magic constant against it.
 * The mirror is never validated only against itself: a hand-written struct padded
 * to the wrong size still "looks right" and mis-loads silently.
 *
 * MEMBER NAMES ARE THE REAL HEADER'S NAMES, VERBATIM (camelCase, not this
 * codebase's snake_case). This is deliberate: renaming each field would insert a
 * translation step between the real header and this mirror, and a translation
 * step is exactly where an ABI error hides. Keeping the names identical lets a
 * reader diff the two definitions line by line.
 *
 * NAMING PREFIX: the mirror keeps an Acl* prefix so it never collides with the
 * aclrt* names of the real header in a TU that includes both.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace tvm::tl::ascendcl {

// ---------------------------------------------------------------------------
// aclrtBinaryLoadOptionType  (real header: typedef'd unscoped enum, sizeof 4)
// ---------------------------------------------------------------------------
enum AclBinaryLoadOptionType {
  kAclBinaryLoadOptLazyLoad = 1,
  kAclBinaryLoadOptLazyMagic = 2,
  kAclBinaryLoadOptMagic = 2,
  kAclBinaryLoadOptCpuKernelMode = 3,
};

// ---------------------------------------------------------------------------
// aclrtBinaryLoadOptionValue  (real header: union, sizeof 16, align 4)
// The widest member is the reserved uint32 array, which is what fixes the size.
// ---------------------------------------------------------------------------
union AclBinaryLoadOptionValue {
  uint32_t isLazyLoad;
  uint32_t magic;
  int32_t cpuKernelMode;
  uint32_t rsv[4];
};

// ---------------------------------------------------------------------------
// aclrtBinaryLoadOption  (real header: struct, sizeof 20, align 4)
// ---------------------------------------------------------------------------
struct AclBinaryLoadOption {
  AclBinaryLoadOptionType type;
  AclBinaryLoadOptionValue value;
};

// ---------------------------------------------------------------------------
// aclrtBinaryLoadOptions  (real header: struct, sizeof 16, align 8)
// ---------------------------------------------------------------------------
struct AclBinaryLoadOptions {
  AclBinaryLoadOption *options;
  size_t numOpt;
};

// ---------------------------------------------------------------------------
// aclrtEngineType  (real header: typedef'd unscoped enum, sizeof 4)
// ---------------------------------------------------------------------------
enum AclEngineType {
  kAclEngineTypeAic = 0,
  kAclEngineTypeAiv = 1,
};

// ---------------------------------------------------------------------------
// aclrtLaunchKernelAttrId  (real header: typedef'd unscoped enum, sizeof 4)
// LOCAL_MEMORY_SIZE is deprecated in the real header and shares value 2 with
// DYN_UBUF_SIZE; both enumerators are mirrored so the values can be asserted.
// ---------------------------------------------------------------------------
enum AclLaunchKernelAttrId {
  kAclLaunchKernelAttrSchemMode = 1,
  kAclLaunchKernelAttrLocalMemorySize = 2,
  kAclLaunchKernelAttrDynUbufSize = 2,
  kAclLaunchKernelAttrEngineType = 3,
  kAclLaunchKernelAttrBlockdimOffset = 4,
  kAclLaunchKernelAttrBlockTaskPrefetch = 5,
  kAclLaunchKernelAttrDataDump = 6,
  kAclLaunchKernelAttrTimeout = 7,
  kAclLaunchKernelAttrTimeoutUs = 8,
};

// ---------------------------------------------------------------------------
// aclrtTimeoutUs  (real header: struct, sizeof 8, align 4)
// ---------------------------------------------------------------------------
struct AclTimeoutUs {
  uint32_t timeoutLow;
  uint32_t timeoutHigh;
};

// ---------------------------------------------------------------------------
// aclrtLaunchKernelAttrValue  (real header: union, sizeof 16, align 4)
// ---------------------------------------------------------------------------
union AclLaunchKernelAttrValue {
  uint8_t schemMode;
  uint32_t localMemorySize;
  uint32_t dynUBufSize;
  AclEngineType engineType;
  uint32_t blockDimOffset;
  uint8_t isBlockTaskPrefetch;
  uint8_t isDataDump;
  uint16_t timeout;
  AclTimeoutUs timeoutUs;
  uint32_t rsv[4];
};

// ---------------------------------------------------------------------------
// aclrtLaunchKernelAttr  (real header: struct, sizeof 20, align 4)
// ---------------------------------------------------------------------------
struct AclLaunchKernelAttr {
  AclLaunchKernelAttrId id;
  AclLaunchKernelAttrValue value;
};

// ---------------------------------------------------------------------------
// aclrtLaunchKernelCfg  (real header: struct, sizeof 16, align 8)
// ---------------------------------------------------------------------------
struct AclLaunchKernelCfg {
  AclLaunchKernelAttr *attrs;
  size_t numAttrs;
};

// ---------------------------------------------------------------------------
// Mirrored magic constants (real header: #define ... U)
// ---------------------------------------------------------------------------
constexpr uint32_t kAclBinaryMagicElfAicore = 0x43554245U;
constexpr uint32_t kAclBinaryMagicElfVectorCore = 0x41415246U;
constexpr uint32_t kAclBinaryMagicElfCubeCore = 0x41494343U;

} // namespace tvm::tl::ascendcl
