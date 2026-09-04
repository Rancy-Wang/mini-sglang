#include <minisgl/utils.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

namespace {

auto _is_1d_cpu_int_tensor(const tvm::ffi::TensorView tensor) -> bool {
  return tensor.ndim() == 1 && tensor.is_contiguous() &&
         tensor.device().device_type == kDLCPU &&
         (tensor.dtype().code == kDLInt) &&
         (tensor.dtype().bits == 32 || tensor.dtype().bits == 64);
}

auto _is_1d_cpu_bool_tensor(const tvm::ffi::TensorView tensor) -> bool {
  return tensor.ndim() == 1 && tensor.is_contiguous() &&
         tensor.device().device_type == kDLCPU &&
         tensor.dtype().code == kDLBool && tensor.dtype().bits == 8;
}

auto _is_radix_record_tensor(const tvm::ffi::TensorView tensor) -> bool {
  return tensor.ndim() == 2 && tensor.size(1) == 4 && tensor.is_contiguous() &&
         tensor.device().device_type == kDLCPU &&
         tensor.dtype().code == kDLInt && tensor.dtype().bits == 32;
}

using RecordCompare = size_t (*)(const int32_t *, const int32_t *, size_t);

auto _compare_records_portable(const int32_t *a, const int32_t *b, size_t rows)
    -> size_t {
  for (size_t row = 0; row < rows; ++row) {
    if (std::memcmp(a + row * 4, b + row * 4, 4 * sizeof(int32_t)) != 0) {
      return row;
    }
  }
  return rows;
}

#if defined(__aarch64__)
auto _compare_records_neon(const int32_t *a, const int32_t *b, size_t rows)
    -> size_t {
  for (size_t row = 0; row < rows; ++row) {
    const auto equal =
        vceqq_s32(vld1q_s32(a + row * 4), vld1q_s32(b + row * 4));
    const auto lanes = vreinterpretq_u64_u32(equal);
    if (vgetq_lane_u64(lanes, 0) != UINT64_MAX ||
        vgetq_lane_u64(lanes, 1) != UINT64_MAX) {
      return row;
    }
  }
  return rows;
}
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) auto
_compare_records_avx2(const int32_t *a, const int32_t *b, size_t rows)
    -> size_t {
  size_t row = 0;
  for (; row + 2 <= rows; row += 2) {
    const auto left =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + row * 4));
    const auto right =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + row * 4));
    if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(left, right)) != -1) {
      return _compare_records_portable(a + row * 4, b + row * 4, 2) + row;
    }
  }
  return _compare_records_portable(a + row * 4, b + row * 4, rows - row) + row;
}

__attribute__((target("avx512f"))) auto
_compare_records_avx512(const int32_t *a, const int32_t *b, size_t rows)
    -> size_t {
  size_t row = 0;
  for (; row + 4 <= rows; row += 4) {
    const auto left = _mm512_loadu_si512(a + row * 4);
    const auto right = _mm512_loadu_si512(b + row * 4);
    if (_mm512_cmpeq_epi32_mask(left, right) != 0xFFFF) {
      return _compare_records_portable(a + row * 4, b + row * 4, 4) + row;
    }
  }
  return _compare_records_portable(a + row * 4, b + row * 4, rows - row) + row;
}
#endif

auto _record_compare() -> RecordCompare {
#if defined(__aarch64__)
  return _compare_records_neon;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx512f")) {
    return _compare_records_avx512;
  }
  if (__builtin_cpu_supports("avx2")) {
    return _compare_records_avx2;
  }
#endif
  return _compare_records_portable;
}

auto radix_record_compare_backend() -> int64_t {
#if defined(__aarch64__)
  return 1;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx512f")) {
    return 3;
  }
  if (__builtin_cpu_supports("avx2")) {
    return 2;
  }
#endif
  return 0;
}

auto radix_record_edge_hash(const tvm::ffi::TensorView records) -> int64_t {
  host::RuntimeCheck(
      _is_radix_record_tensor(records) && records.size(0) > 0,
      "Radix edge hashing requires a non-empty int32 [N, 4] tensor.");
  const auto *record = static_cast<const int32_t *>(records.data_ptr());
  const size_t rows = record[0] == 1 ? records.size(0) : 1;
  uint64_t hash = 1469598103934665603ULL;
  size_t edge_rows = 0;
  while (edge_rows < rows && (edge_rows == 0 || record[edge_rows * 4] == 1)) {
    for (size_t field = 0; field < 4; ++field) {
      hash ^= static_cast<uint32_t>(record[edge_rows * 4 + field]);
      hash *= 1099511628211ULL;
    }
    ++edge_rows;
  }
  hash ^= edge_rows;
  hash *= 1099511628211ULL;
  return static_cast<int64_t>(hash & INT64_MAX);
}

auto radix_record_edge_equal(const tvm::ffi::TensorView a,
                             const tvm::ffi::TensorView b) -> bool {
  host::RuntimeCheck(
      _is_radix_record_tensor(a) && a.size(0) > 0 &&
          _is_radix_record_tensor(b) && b.size(0) > 0,
      "Radix edge comparison requires non-empty int32 [N, 4] tensors.");
  const auto *a_ptr = static_cast<const int32_t *>(a.data_ptr());
  const auto *b_ptr = static_cast<const int32_t *>(b.data_ptr());
  size_t a_rows = 1;
  size_t b_rows = 1;
  if (a_ptr[0] == 1) {
    while (a_rows < a.size(0) && a_ptr[a_rows * 4] == 1) ++a_rows;
  }
  if (b_ptr[0] == 1) {
    while (b_rows < b.size(0) && b_ptr[b_rows * 4] == 1) ++b_rows;
  }
  return a_rows == b_rows &&
         std::memcmp(a_ptr, b_ptr, a_rows * 4 * sizeof(int32_t)) == 0;
}

auto radix_record_retry_token(const tvm::ffi::TensorView records) -> int64_t {
  host::RuntimeCheck(
      _is_radix_record_tensor(records) && records.size(0) > 0,
      "Retry edge lookup requires a non-empty int32 [N, 4] tensor.");
  const auto *record = static_cast<const int32_t *>(records.data_ptr());
  return record[0] == 0 ? record[1] : -1;
}

auto fast_compare_key(const tvm::ffi::TensorView a,
                      const tvm::ffi::TensorView b) -> size_t {
  host::RuntimeCheck(_is_1d_cpu_int_tensor(a) && _is_1d_cpu_int_tensor(b),
                     "Both tensors must be 1D CPU int tensors.");
  host::RuntimeCheck(a.dtype() == b.dtype());
  const auto a_ptr = a.data_ptr();
  const auto b_ptr = b.data_ptr();
  const auto common_len = std::min(a.size(0), b.size(0));
  if (a.dtype().bits == 64) {
    const auto a_ptr_64 = static_cast<const int64_t *>(a_ptr);
    const auto b_ptr_64 = static_cast<const int64_t *>(b_ptr);
    const auto diff_pos =
        std::mismatch(a_ptr_64, a_ptr_64 + common_len, b_ptr_64);
    return static_cast<size_t>(diff_pos.first - a_ptr_64);
  } else {
    const auto a_ptr_32 = static_cast<const int32_t *>(a_ptr);
    const auto b_ptr_32 = static_cast<const int32_t *>(b_ptr);
    const auto diff_pos =
        std::mismatch(a_ptr_32, a_ptr_32 + common_len, b_ptr_32);
    return static_cast<size_t>(diff_pos.first - a_ptr_32);
  }
}

auto fast_compare_radix_key(const tvm::ffi::TensorView a,
                            const tvm::ffi::TensorView b,
                            const tvm::ffi::TensorView a_virtual,
                            const tvm::ffi::TensorView b_virtual) -> size_t {
  host::RuntimeCheck(_is_1d_cpu_int_tensor(a) && _is_1d_cpu_int_tensor(b),
                     "Both Radix key tensors must be 1D CPU int tensors.");
  host::RuntimeCheck(_is_1d_cpu_bool_tensor(a_virtual) &&
                         _is_1d_cpu_bool_tensor(b_virtual),
                     "Both virtual masks must be 1D CPU bool tensors.");
  host::RuntimeCheck(a.dtype() == b.dtype(),
                     "Both Radix key tensors must have the same dtype.");
  host::RuntimeCheck(
      a.size(0) == a_virtual.size(0) && b.size(0) == b_virtual.size(0),
      "Each Radix key tensor must match its virtual mask length.");

  const auto common_len = std::min(a.size(0), b.size(0));
  const auto a_virtual_ptr = static_cast<const bool *>(a_virtual.data_ptr());
  const auto b_virtual_ptr = static_cast<const bool *>(b_virtual.data_ptr());
  if (a.dtype().bits == 64) {
    const auto a_ptr = static_cast<const int64_t *>(a.data_ptr());
    const auto b_ptr = static_cast<const int64_t *>(b.data_ptr());
    for (size_t idx = 0; idx < common_len; ++idx) {
      if (a_ptr[idx] != b_ptr[idx] ||
          a_virtual_ptr[idx] != b_virtual_ptr[idx]) {
        return idx;
      }
    }
  } else {
    const auto a_ptr = static_cast<const int32_t *>(a.data_ptr());
    const auto b_ptr = static_cast<const int32_t *>(b.data_ptr());
    for (size_t idx = 0; idx < common_len; ++idx) {
      if (a_ptr[idx] != b_ptr[idx] ||
          a_virtual_ptr[idx] != b_virtual_ptr[idx]) {
        return idx;
      }
    }
  }
  return common_len;
}

auto fast_compare_radix_records(const tvm::ffi::TensorView a,
                                const tvm::ffi::TensorView b) -> size_t {
  host::RuntimeCheck(
      _is_radix_record_tensor(a) && _is_radix_record_tensor(b),
      "Both Radix records must be contiguous CPU int32 [N, 4] tensors.");
  const auto common_len = std::min(a.size(0), b.size(0));
  const auto *a_ptr = static_cast<const int32_t *>(a.data_ptr());
  const auto *b_ptr = static_cast<const int32_t *>(b.data_ptr());
  static const auto compare = _record_compare();
  return compare(a_ptr, b_ptr, common_len);
}

auto _retry_record_equal(const int32_t *cached, const int32_t *target) -> bool {
  if (std::memcmp(cached, target, 2 * sizeof(int32_t)) != 0) {
    return false;
  }
  return cached[0] == 0 ||
         std::memcmp(cached + 2, target + 2, 2 * sizeof(int32_t)) == 0;
}

auto fast_compare_retry_radix_records(const tvm::ffi::TensorView cached,
                                      const tvm::ffi::TensorView target)
    -> size_t {
  host::RuntimeCheck(
      _is_radix_record_tensor(cached) && _is_radix_record_tensor(target),
      "Both Retry Radix records must be contiguous CPU int32 [N, 4] tensors.");
  const auto common_len = std::min(cached.size(0), target.size(0));
  const auto *cached_ptr = static_cast<const int32_t *>(cached.data_ptr());
  const auto *target_ptr = static_cast<const int32_t *>(target.data_ptr());
  for (size_t row = 0; row < common_len; ++row) {
    if (!_retry_record_equal(cached_ptr + row * 4, target_ptr + row * 4)) {
      return row;
    }
  }
  return common_len;
}

auto fast_compare_retry_radix_records_plan(
    const tvm::ffi::TensorView cached, const tvm::ffi::TensorView target,
    const tvm::ffi::TensorView cached_key_to_token,
    const tvm::ffi::TensorView target_key_to_token,
    const tvm::ffi::TensorView output_plan, const tvm::ffi::TensorView status)
    -> void {
  host::RuntimeCheck(
      _is_radix_record_tensor(cached) && _is_radix_record_tensor(target),
      "Both Retry Radix records must be contiguous CPU int32 [N, 4] tensors.");
  host::RuntimeCheck(_is_1d_cpu_int_tensor(cached_key_to_token) &&
                         cached_key_to_token.dtype().bits == 64 &&
                         _is_1d_cpu_int_tensor(target_key_to_token) &&
                         target_key_to_token.dtype().bits == 64,
                     "Retry key-to-token mappings must be CPU int64 vectors.");
  host::RuntimeCheck(_is_radix_record_tensor(output_plan) &&
                         _is_1d_cpu_int_tensor(status) &&
                         status.dtype().bits == 64 && status.size(0) >= 2,
                     "Retry output plan/status tensors have invalid layouts.");
  const auto common_len = std::min(cached.size(0), target.size(0));
  host::RuntimeCheck(
      cached_key_to_token.size(0) >= common_len &&
          target_key_to_token.size(0) >= common_len &&
          output_plan.size(0) >= common_len,
      "Retry plan capacity does not cover the common record prefix.");

  const auto *cached_ptr = static_cast<const int32_t *>(cached.data_ptr());
  const auto *target_ptr = static_cast<const int32_t *>(target.data_ptr());
  const auto *cached_map =
      static_cast<const int64_t *>(cached_key_to_token.data_ptr());
  const auto *target_map =
      static_cast<const int64_t *>(target_key_to_token.data_ptr());
  auto *plan = static_cast<int32_t *>(output_plan.data_ptr());
  auto *result = static_cast<int64_t *>(status.data_ptr());
  size_t row = 0;
  size_t plan_count = 0;
  for (; row < common_len; ++row) {
    const auto *cached_record = cached_ptr + row * 4;
    const auto *target_record = target_ptr + row * 4;
    if (!_retry_record_equal(cached_record, target_record)) {
      break;
    }
    if (cached_record[0] != 0 || cached_record[3] == target_record[3]) {
      continue;
    }
    host::RuntimeCheck(cached_map[row] >= 0 && cached_map[row] <= INT32_MAX &&
                           target_map[row] >= 0 && target_map[row] <= INT32_MAX,
                       "Retry token mapping exceeds the int32 plan range.");
    plan[plan_count * 4] = static_cast<int32_t>(cached_map[row]);
    plan[plan_count * 4 + 1] = static_cast<int32_t>(target_map[row]);
    plan[plan_count * 4 + 2] = cached_record[3];
    plan[plan_count * 4 + 3] = target_record[3];
    ++plan_count;
  }
  result[0] = row;
  result[1] = plan_count;
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_compare_key, fast_compare_key);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_compare_radix_key, fast_compare_radix_key);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_compare_radix_records,
                              fast_compare_radix_records);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_compare_retry_radix_records,
                              fast_compare_retry_radix_records);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_compare_retry_radix_records_plan,
                              fast_compare_retry_radix_records_plan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(radix_record_compare_backend,
                              radix_record_compare_backend);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(radix_record_edge_hash, radix_record_edge_hash);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(radix_record_edge_equal, radix_record_edge_equal);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(radix_record_retry_token,
                              radix_record_retry_token);
