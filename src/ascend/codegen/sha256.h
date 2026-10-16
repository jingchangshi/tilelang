/*!
 * \file sha256.h
 * \brief Small self-contained SHA-256 (FIPS 180-4) for the PREBUILT binary
 *        identity check.
 *
 * WHY THIS EXISTS
 * The PREBUILT HIVM entry must refuse any device binary whose sha256 is not the
 * frozen P3 digest (goal section 3: "任何 Binary SHA 不匹配必须拒绝"). The
 * digest has to be computed INSIDE the process that reads the bytes, because
 * goal CR-ID-2 requires the bytes that are hashed to be the bytes that are
 * loaded -- re-opening the path between the check and the use is invalid, and
 * an out-of-process `sha256sum` cannot satisfy that.
 *
 * WHY NOT A LIBRARY
 * OpenSSL is NOT available in the build container (no <openssl/sha.h>), and the
 * tree vendors no hash implementation. TileLang also keeps no CANN or extra
 * third-party build dependency. So this is a deliberately small, dependency-free
 * implementation.
 *
 * VERIFICATION
 * This is NOT validated by comparing it against itself. The independent test at
 * hivm-poc-evidence/p4-20261016/impl/p4b-prebuilt/test_prebuilt_builder.cc
 * carries a SEPARATE SHA-256 written from the same specification and asserts
 * both against the NIST/FIPS 180-4 vectors; the expected digest for the frozen
 * binary is produced by coreutils `sha256sum`, a third instrument.
 */

#ifndef TVM_TL_ASCEND_CODEGEN_SHA256_H_
#define TVM_TL_ASCEND_CODEGEN_SHA256_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace tvm {
namespace codegen {
namespace hivm_prebuilt {

class Sha256 {
public:
  static constexpr size_t kDigestBytes = 32;

  Sha256() { Reset(); }

  void Reset() {
    state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
              0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    buffer_.fill(0);
    buffered_ = 0;
    total_bits_ = 0;
  }

  void Update(const uint8_t *data, size_t len) {
    if (len != 0) {
      total_bits_ += static_cast<uint64_t>(len) * 8U;
    }
    while (len > 0) {
      const size_t take = std::min(len, buffer_.size() - buffered_);
      std::memcpy(buffer_.data() + buffered_, data, take);
      buffered_ += take;
      data += take;
      len -= take;
      if (buffered_ == buffer_.size()) {
        Transform(buffer_.data());
        buffered_ = 0;
      }
    }
  }

  void Update(const std::string &data) {
    Update(reinterpret_cast<const uint8_t *>(data.data()), data.size());
  }

  // Finalize and return the lowercase hex digest. The object must be Reset()
  // before it is reused.
  std::string HexDigest() {
    const uint64_t total_bits = total_bits_;
    const uint8_t pad = 0x80;
    Update(&pad, 1);
    const uint8_t zero = 0x00;
    while (buffered_ != 56) {
      Update(&zero, 1);
    }
    uint8_t length_be[8];
    for (int i = 0; i < 8; ++i) {
      length_be[i] = static_cast<uint8_t>(total_bits >> (56 - 8 * i));
    }
    Update(length_be, sizeof(length_be));

    static const char *kHex = "0123456789abcdef";
    std::string out;
    out.reserve(kDigestBytes * 2);
    for (uint32_t word : state_) {
      for (int shift = 28; shift >= 0; shift -= 4) {
        out.push_back(kHex[(word >> shift) & 0xfU]);
      }
    }
    return out;
  }

private:
  static uint32_t Rotr(uint32_t value, uint32_t amount) {
    return (value >> amount) | (value << (32U - amount));
  }

  static const std::array<uint32_t, 64> &K() {
    static const std::array<uint32_t, 64> k = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
        0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
        0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
        0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
        0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
        0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
        0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
        0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
        0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    return k;
  }

  void Transform(const uint8_t *block) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
             (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
             (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
             static_cast<uint32_t>(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t temp1 = h + s1 + ch + K()[i] + w[i];
      const uint32_t s0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<uint32_t, 8> state_{};
  std::array<uint8_t, 64> buffer_{};
  size_t buffered_{0};
  uint64_t total_bits_{0};
};

inline std::string Sha256Hex(const void *data, size_t size) {
  Sha256 hasher;
  hasher.Update(reinterpret_cast<const uint8_t *>(data), size);
  return hasher.HexDigest();
}

inline std::string Sha256Hex(const std::string &data) {
  return Sha256Hex(data.data(), data.size());
}

} // namespace hivm_prebuilt
} // namespace codegen
} // namespace tvm

#endif // TVM_TL_ASCEND_CODEGEN_SHA256_H_
