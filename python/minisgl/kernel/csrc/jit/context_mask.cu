#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

struct ContextMaskKernelParams {
  const std::int32_t *__restrict__ token_visible_until;
  std::uint8_t *__restrict__ output;
  std::int64_t query_start;
  std::int64_t query_length;
  std::int64_t key_length;
  std::int64_t num_bits;
  std::int64_t num_bytes;
};

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void
build_context_mask(const __grid_constant__ ContextMaskKernelParams params) {
  using namespace device;

  const auto byte_idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  PDL::wait<kUsePDL>();
  if (byte_idx < params.num_bytes) {
    std::uint8_t packed = 0;
#pragma unroll
    for (std::int64_t bit = 0; bit < 8; ++bit) {
      const auto flat_idx = byte_idx * 8 + bit;
      if (flat_idx >= params.num_bits) {
        break;
      }
      const auto query_offset = flat_idx / params.key_length;
      const auto key_position = flat_idx - query_offset * params.key_length;
      const auto query_position = params.query_start + query_offset;
      const bool causal = key_position <= query_position;
      const bool visible =
          query_position < params.token_visible_until[key_position];
      packed |= static_cast<std::uint8_t>(
          (causal && visible) ? (1u << bit) : 0u);
    }
    params.output[byte_idx] = packed;
  }
  PDL::launch<kUsePDL>();
}

template <std::size_t num_threads = 256,
          std::size_t max_concurrency = 1,
          bool use_pdl = false>
struct ContextMaskKernel {
  static void run(const tvm::ffi::TensorView token_visible_until,
                  const tvm::ffi::TensorView output,
                  std::int64_t query_start,
                  std::int64_t query_length,
                  std::int64_t key_length) {
    using namespace host;
    auto full_length = SymbolicSize{"full_length"};
    auto packed_length = SymbolicSize{"packed_length"};
    auto device = SymbolicDevice{};

    TensorMatcher({full_length})
        .with_dtype<std::int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(token_visible_until);
    TensorMatcher({packed_length})
        .with_dtype<std::uint8_t>()
        .with_device<kDLCUDA>(device)
        .verify(output);

    RuntimeCheck(query_start >= 0 && query_length > 0 && key_length > 0,
                 "ContextMaskKernel: invalid query/key range.");
    RuntimeCheck(query_start + query_length <= full_length.unwrap(),
                 "ContextMaskKernel: query range exceeds metadata length.");
    RuntimeCheck(key_length <= full_length.unwrap(),
                 "ContextMaskKernel: key range exceeds metadata length.");

    const auto num_bits = query_length * key_length;
    const auto num_bytes = (num_bits + 7) / 8;
    RuntimeCheck(packed_length.unwrap() == num_bytes,
                 "ContextMaskKernel: packed output length mismatch.");

    const auto params = ContextMaskKernelParams{
        .token_visible_until = static_cast<const std::int32_t *>(
            token_visible_until.data_ptr()),
        .output = static_cast<std::uint8_t *>(output.data_ptr()),
        .query_start = query_start,
        .query_length = query_length,
        .key_length = key_length,
        .num_bits = num_bits,
        .num_bytes = num_bytes,
    };
    const auto num_blocks =
        div_ceil(num_bytes, static_cast<std::int64_t>(num_threads));
    LaunchKernel(num_blocks, num_threads, device.unwrap())
        .with_attr(use_pdl)(
            build_context_mask<num_threads, max_concurrency, use_pdl>, params);
  }
};

struct ContextMaskUnpackedKernelParams {
  const std::int32_t *__restrict__ token_visible_until;
  std::uint8_t *__restrict__ output;
  std::int64_t query_start;
  std::int64_t key_length;
  std::int64_t num_elements;
};

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void
build_context_mask_unpacked(
    const __grid_constant__ ContextMaskUnpackedKernelParams params) {
  using namespace device;

  const auto flat_idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  PDL::wait<kUsePDL>();
  if (flat_idx < params.num_elements) {
    const auto query_offset = flat_idx / params.key_length;
    const auto key_position = flat_idx - query_offset * params.key_length;
    const auto query_position = params.query_start + query_offset;
    const bool causal = key_position <= query_position;
    const bool visible =
        query_position < params.token_visible_until[key_position];
    params.output[flat_idx] = static_cast<std::uint8_t>(causal && visible);
  }
  PDL::launch<kUsePDL>();
}

template <std::size_t num_threads = 256,
          std::size_t max_concurrency = 1,
          bool use_pdl = false>
struct ContextMaskUnpackedKernel {
  static void run(const tvm::ffi::TensorView token_visible_until,
                  const tvm::ffi::TensorView output,
                  std::int64_t query_start,
                  std::int64_t query_length,
                  std::int64_t key_length) {
    using namespace host;
    auto full_length = SymbolicSize{"full_length"};
    auto output_length = SymbolicSize{"output_length"};
    auto device = SymbolicDevice{};

    TensorMatcher({full_length})
        .with_dtype<std::int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(token_visible_until);
    TensorMatcher({output_length})
        .with_dtype<std::uint8_t>()
        .with_device<kDLCUDA>(device)
        .verify(output);

    RuntimeCheck(query_start >= 0 && query_length > 0 && key_length > 0,
                 "ContextMaskUnpackedKernel: invalid query/key range.");
    RuntimeCheck(query_start + query_length <= full_length.unwrap(),
                 "ContextMaskUnpackedKernel: query range exceeds metadata length.");
    RuntimeCheck(key_length <= full_length.unwrap(),
                 "ContextMaskUnpackedKernel: key range exceeds metadata length.");

    const auto num_elements = query_length * key_length;
    RuntimeCheck(output_length.unwrap() == num_elements,
                 "ContextMaskUnpackedKernel: output length mismatch.");

    const auto params = ContextMaskUnpackedKernelParams{
        .token_visible_until = static_cast<const std::int32_t *>(
            token_visible_until.data_ptr()),
        .output = static_cast<std::uint8_t *>(output.data_ptr()),
        .query_start = query_start,
        .key_length = key_length,
        .num_elements = num_elements,
    };
    const auto num_blocks =
        div_ceil(num_elements, static_cast<std::int64_t>(num_threads));
    LaunchKernel(num_blocks, num_threads, device.unwrap())
        .with_attr(use_pdl)(
            build_context_mask_unpacked<num_threads, max_concurrency, use_pdl>,
            params);
  }
};

} // namespace
