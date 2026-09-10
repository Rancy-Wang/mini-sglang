#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <cstdint>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

namespace {

auto is_cuda_int32_vector(const tvm::ffi::TensorView tensor) -> bool {
  return tensor.ndim() == 1 && tensor.is_contiguous() &&
         tensor.device().device_type == kDLCUDA &&
         tensor.dtype().code == kDLInt && tensor.dtype().bits == 32;
}

auto same_device(const tvm::ffi::TensorView lhs,
                 const tvm::ffi::TensorView rhs) -> bool {
  return lhs.device().device_type == rhs.device().device_type &&
         lhs.device().device_id == rhs.device().device_id;
}

auto same_dtype(const tvm::ffi::TensorView lhs,
                const tvm::ffi::TensorView rhs) -> bool {
  return lhs.dtype().code == rhs.dtype().code &&
         lhs.dtype().bits == rhs.dtype().bits &&
         lhs.dtype().lanes == rhs.dtype().lanes;
}

template <typename T, bool kDirect, bool kWriteFlat, bool kWritePadded,
          unsigned kBlockSize>
__global__ __launch_bounds__(kBlockSize) void compile_page_table_kernel(
    const T *__restrict__ source,
    const int32_t *__restrict__ segment_table_indices,
    const int32_t *__restrict__ key_positions,
    const int32_t *__restrict__ key_offsets, T *__restrict__ flat_indices,
    T *__restrict__ padded_page_table, int64_t source_stride,
    int64_t max_seqlen_k) {
  const int64_t segment = blockIdx.x;
  const int64_t local_key =
      static_cast<int64_t>(blockIdx.y) * kBlockSize + threadIdx.x;
  if (local_key >= max_seqlen_k) return;

  const int64_t key_start = key_offsets[segment];
  const int64_t key_count = key_offsets[segment + 1] - key_start;
  const bool valid = local_key < key_count;
  T page = 0;
  if (valid) {
    const int64_t key_position = key_positions[key_start + local_key];
    if constexpr (kDirect) {
      page = source[key_position];
    } else {
      const int64_t table = segment_table_indices[segment];
      page = source[table * source_stride + key_position];
    }
    if constexpr (kWriteFlat) {
      flat_indices[key_start + local_key] = page;
    }
  }
  if constexpr (kWritePadded) {
    padded_page_table[segment * max_seqlen_k + local_key] = page;
  }
}

template <bool kDirect, bool kWriteFlat, bool kWritePadded,
          unsigned kBlockSize, typename T>
auto launch_typed(const tvm::ffi::TensorView source,
                  const tvm::ffi::TensorView *segment_table_indices,
                  const tvm::ffi::TensorView key_positions,
                  const tvm::ffi::TensorView key_offsets,
                  const tvm::ffi::TensorView *flat_indices,
                  const tvm::ffi::TensorView *padded_page_table,
                  int64_t max_seqlen_k) -> void {
  const int64_t num_segments = key_offsets.size(0) - 1;
  const int64_t source_stride = kDirect ? 0 : source.stride(0);
  const auto *table_ptr =
      kDirect ? nullptr
              : static_cast<const int32_t *>(segment_table_indices->data_ptr());
  auto *flat_ptr = static_cast<T *>(nullptr);
  auto *padded_ptr = static_cast<T *>(nullptr);
  if constexpr (kWriteFlat) {
    flat_ptr = static_cast<T *>(flat_indices->data_ptr());
  }
  if constexpr (kWritePadded) {
    padded_ptr = static_cast<T *>(padded_page_table->data_ptr());
  }
  const auto kernel =
      compile_page_table_kernel<T, kDirect, kWriteFlat, kWritePadded,
                                kBlockSize>;
  const auto blocks_per_segment =
      static_cast<unsigned>((max_seqlen_k + kBlockSize - 1) / kBlockSize);
  host::LaunchKernel(
      dim3(static_cast<unsigned>(num_segments), blocks_per_segment),
      dim3(kBlockSize), source.device())(
      kernel, static_cast<const T *>(source.data_ptr()), table_ptr,
      static_cast<const int32_t *>(key_positions.data_ptr()),
      static_cast<const int32_t *>(key_offsets.data_ptr()), flat_ptr,
      padded_ptr, source_stride, max_seqlen_k);
}

template <bool kDirect, bool kWriteFlat, bool kWritePadded,
          unsigned kBlockSize>
auto launch_by_dtype(const tvm::ffi::TensorView source,
                     const tvm::ffi::TensorView *segment_table_indices,
                     const tvm::ffi::TensorView key_positions,
                     const tvm::ffi::TensorView key_offsets,
                     const tvm::ffi::TensorView *flat_indices,
                     const tvm::ffi::TensorView *padded_page_table,
                     int64_t max_seqlen_k) -> void {
  if (source.dtype().bits == 32) {
    launch_typed<kDirect, kWriteFlat, kWritePadded, kBlockSize, int32_t>(
        source, segment_table_indices, key_positions, key_offsets, flat_indices,
        padded_page_table, max_seqlen_k);
  } else {
    launch_typed<kDirect, kWriteFlat, kWritePadded, kBlockSize, int64_t>(
        source, segment_table_indices, key_positions, key_offsets, flat_indices,
        padded_page_table, max_seqlen_k);
  }
}

template <bool kDirect, bool kWriteFlat, bool kWritePadded>
auto validate_and_launch(
    const tvm::ffi::TensorView source,
    const tvm::ffi::TensorView *segment_table_indices,
    const tvm::ffi::TensorView key_positions,
    const tvm::ffi::TensorView key_offsets, int64_t max_seqlen_k,
    int64_t block_size, const tvm::ffi::TensorView *flat_indices,
    const tvm::ffi::TensorView *padded_page_table) -> void {
  host::RuntimeCheck(source.is_contiguous() &&
                         source.device().device_type == kDLCUDA &&
                         source.dtype().code == kDLInt &&
                         (source.dtype().bits == 32 ||
                          source.dtype().bits == 64) &&
                         source.dtype().lanes == 1,
                     "Context page source must be contiguous CUDA int32/int64");
  host::RuntimeCheck(source.ndim() == (kDirect ? 1 : 2),
                     "Context page source has an invalid rank");
  host::RuntimeCheck(is_cuda_int32_vector(key_positions) &&
                         is_cuda_int32_vector(key_offsets) &&
                         same_device(source, key_positions) &&
                         same_device(source, key_offsets),
                     "Context key metadata must be CUDA int32 vectors on the source device");
  const int64_t num_segments = key_offsets.size(0) - 1;
  host::RuntimeCheck(num_segments > 0 && max_seqlen_k > 0,
                     "Context page-table dimensions must be positive");

  if constexpr (!kDirect) {
    host::RuntimeCheck(segment_table_indices != nullptr &&
                           is_cuda_int32_vector(*segment_table_indices) &&
                           segment_table_indices->size(0) == num_segments &&
                           same_device(source, *segment_table_indices),
                       "Context segment ownership must be a matching CUDA int32 vector");
  }
  if constexpr (kWriteFlat) {
    host::RuntimeCheck(flat_indices != nullptr && flat_indices->ndim() == 1 &&
                           flat_indices->is_contiguous() &&
                           flat_indices->size(0) == key_positions.size(0) &&
                           same_device(source, *flat_indices) &&
                           same_dtype(source, *flat_indices),
                       "Context flat output has an invalid layout");
  }
  if constexpr (kWritePadded) {
    host::RuntimeCheck(
        padded_page_table != nullptr && padded_page_table->ndim() == 2 &&
            padded_page_table->is_contiguous() &&
            padded_page_table->size(0) == num_segments &&
            padded_page_table->size(1) == max_seqlen_k &&
            same_device(source, *padded_page_table) &&
            same_dtype(source, *padded_page_table),
        "Context padded output has an invalid layout");
  }

  switch (block_size) {
  case 128:
    launch_by_dtype<kDirect, kWriteFlat, kWritePadded, 128>(
        source, segment_table_indices, key_positions, key_offsets, flat_indices,
        padded_page_table, max_seqlen_k);
    break;
  case 256:
    launch_by_dtype<kDirect, kWriteFlat, kWritePadded, 256>(
        source, segment_table_indices, key_positions, key_offsets, flat_indices,
        padded_page_table, max_seqlen_k);
    break;
  case 512:
    launch_by_dtype<kDirect, kWriteFlat, kWritePadded, 512>(
        source, segment_table_indices, key_positions, key_offsets, flat_indices,
        padded_page_table, max_seqlen_k);
    break;
  default:
    host::RuntimeCheck(false,
                       "Context page-table block size must be 128, 256, or 512");
  }
}

auto compile_context_page_table_flat(
    const tvm::ffi::TensorView source,
    const tvm::ffi::TensorView segment_table_indices,
    const tvm::ffi::TensorView key_positions,
    const tvm::ffi::TensorView key_offsets, int64_t max_seqlen_k,
    int64_t block_size, const tvm::ffi::TensorView flat_indices) -> void {
  validate_and_launch<false, true, false>(
      source, &segment_table_indices, key_positions, key_offsets,
      max_seqlen_k, block_size, &flat_indices, nullptr);
}

auto compile_context_page_table_padded(
    const tvm::ffi::TensorView source,
    const tvm::ffi::TensorView segment_table_indices,
    const tvm::ffi::TensorView key_positions,
    const tvm::ffi::TensorView key_offsets, int64_t max_seqlen_k,
    int64_t block_size,
    const tvm::ffi::TensorView padded_page_table) -> void {
  validate_and_launch<false, false, true>(
      source, &segment_table_indices, key_positions, key_offsets,
      max_seqlen_k, block_size, nullptr, &padded_page_table);
}

auto compile_context_page_table_both(
    const tvm::ffi::TensorView source,
    const tvm::ffi::TensorView segment_table_indices,
    const tvm::ffi::TensorView key_positions,
    const tvm::ffi::TensorView key_offsets, int64_t max_seqlen_k,
    int64_t block_size, const tvm::ffi::TensorView flat_indices,
    const tvm::ffi::TensorView padded_page_table) -> void {
  validate_and_launch<false, true, true>(
      source, &segment_table_indices, key_positions, key_offsets,
      max_seqlen_k, block_size, &flat_indices, &padded_page_table);
}

auto compile_direct_page_table_flat(
    const tvm::ffi::TensorView source,
    const tvm::ffi::TensorView key_positions,
    const tvm::ffi::TensorView key_offsets, int64_t max_seqlen_k,
    int64_t block_size, const tvm::ffi::TensorView flat_indices) -> void {
  validate_and_launch<true, true, false>(
      source, nullptr, key_positions, key_offsets, max_seqlen_k, block_size,
      &flat_indices, nullptr);
}

auto compile_direct_page_table_padded(
    const tvm::ffi::TensorView source,
    const tvm::ffi::TensorView key_positions,
    const tvm::ffi::TensorView key_offsets, int64_t max_seqlen_k,
    int64_t block_size,
    const tvm::ffi::TensorView padded_page_table) -> void {
  validate_and_launch<true, false, true>(
      source, nullptr, key_positions, key_offsets, max_seqlen_k, block_size,
      nullptr, &padded_page_table);
}

auto compile_direct_page_table_both(
    const tvm::ffi::TensorView source,
    const tvm::ffi::TensorView key_positions,
    const tvm::ffi::TensorView key_offsets, int64_t max_seqlen_k,
    int64_t block_size, const tvm::ffi::TensorView flat_indices,
    const tvm::ffi::TensorView padded_page_table) -> void {
  validate_and_launch<true, true, true>(
      source, nullptr, key_positions, key_offsets, max_seqlen_k, block_size,
      &flat_indices, &padded_page_table);
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(compile_context_page_table_flat,
                              compile_context_page_table_flat);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(compile_context_page_table_padded,
                              compile_context_page_table_padded);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(compile_context_page_table_both,
                              compile_context_page_table_both);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(compile_direct_page_table_flat,
                              compile_direct_page_table_flat);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(compile_direct_page_table_padded,
                              compile_direct_page_table_padded);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(compile_direct_page_table_both,
                              compile_direct_page_table_both);
