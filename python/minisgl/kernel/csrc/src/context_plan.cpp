#include <minisgl/utils.h>

#include <algorithm>
#include <cstdint>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

namespace {

auto is_cpu_int32_vector(const tvm::ffi::TensorView tensor) -> bool {
  return tensor.ndim() == 1 && tensor.is_contiguous() &&
         tensor.device().device_type == kDLCPU &&
         tensor.dtype().code == kDLInt && tensor.dtype().bits == 32;
}

auto first_mask_free_conflict_event(
    const tvm::ffi::TensorView active_positions,
    const tvm::ffi::TensorView event_positions,
    const tvm::ffi::TensorView range_offsets,
    const tvm::ffi::TensorView position_ranges, int64_t active_cached_len,
    int64_t effective_event_count) -> int64_t {
  host::RuntimeCheck(is_cpu_int32_vector(active_positions),
                     "active_positions must be a contiguous CPU int32 vector");
  host::RuntimeCheck(is_cpu_int32_vector(event_positions),
                     "event_positions must be a contiguous CPU int32 vector");
  host::RuntimeCheck(is_cpu_int32_vector(range_offsets),
                     "range_offsets must be a contiguous CPU int32 vector");
  host::RuntimeCheck(is_cpu_int32_vector(position_ranges),
                     "position_ranges must be a contiguous CPU int32 vector");
  host::RuntimeCheck(active_cached_len >= 0 &&
                         active_cached_len < active_positions.size(0),
                     "active_cached_len must leave at least one active query");
  host::RuntimeCheck(effective_event_count >= 0 &&
                         effective_event_count <= event_positions.size(0),
                     "effective_event_count is outside event_positions");
  host::RuntimeCheck(range_offsets.size(0) == event_positions.size(0) + 1,
                     "range_offsets must have event_count + 1 entries");
  host::RuntimeCheck(position_ranges.size(0) % 2 == 0,
                     "position_ranges must contain start/end pairs");

  const auto *active =
      static_cast<const int32_t *>(active_positions.data_ptr());
  const auto *events =
      static_cast<const int32_t *>(event_positions.data_ptr());
  const auto *offsets =
      static_cast<const int32_t *>(range_offsets.data_ptr());
  const auto *ranges =
      static_cast<const int32_t *>(position_ranges.data_ptr());
  const int64_t range_count = position_ranges.size(0) / 2;
  host::RuntimeCheck(offsets[0] == 0 &&
                         offsets[event_positions.size(0)] == range_count,
                     "range_offsets does not cover position_ranges");
  for (int64_t event_idx = 0; event_idx < event_positions.size(0);
       ++event_idx) {
    host::RuntimeCheck(offsets[event_idx] <= offsets[event_idx + 1],
                       "range_offsets must be monotonic");
    if (event_idx > 0) {
      host::RuntimeCheck(events[event_idx - 1] < events[event_idx],
                         "event_positions must be strictly increasing");
    }
  }

  const auto *query_begin = active + active_cached_len;
  const auto *query_end = active + active_positions.size(0);
  for (int64_t event_idx = 0; event_idx < effective_event_count;
       ++event_idx) {
    const int64_t range_begin = offsets[event_idx];
    const int64_t range_end = offsets[event_idx + 1];
    host::RuntimeCheck(range_begin >= 0 && range_begin < range_end &&
                           range_end <= range_count,
                       "each effective Drop event must own a non-empty range set");
    const int32_t first_dropped_position = ranges[2 * range_begin];
    const int32_t event_position = events[event_idx];
    int32_t previous_end = -1;
    for (int64_t range_idx = range_begin; range_idx < range_end; ++range_idx) {
      const int32_t start = ranges[2 * range_idx];
      const int32_t end = ranges[2 * range_idx + 1];
      host::RuntimeCheck(start >= 0 && start < end && end <= event_position,
                         "an effective Drop range is outside its event boundary");
      host::RuntimeCheck(range_idx == range_begin || previous_end < start,
                         "effective Drop ranges must be canonical and ordered");
      previous_end = end;
    }

    const auto *query = std::lower_bound(
        query_begin, query_end, first_dropped_position);
    if (query != query_end && *query < event_position) {
      return event_idx;
    }
  }
  return -1;
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(first_mask_free_conflict_event,
                              first_mask_free_conflict_event);
