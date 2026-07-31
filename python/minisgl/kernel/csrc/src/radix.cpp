#include <minisgl/utils.h>

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
         tensor.dtype().code == kDLBool && tensor.dtype().bits == 1;
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
  host::RuntimeCheck(a.size(0) == a_virtual.size(0) &&
                         b.size(0) == b_virtual.size(0),
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

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_compare_key, fast_compare_key);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_compare_radix_key,
                              fast_compare_radix_key);
