#include <minisgl/utils.h>

#include <algorithm>
#include <cstdint>
#include <limits>

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

auto is_cpu_int64_vector(const tvm::ffi::TensorView tensor) -> bool {
  return tensor.ndim() == 1 && tensor.is_contiguous() &&
         tensor.device().device_type == kDLCPU &&
         tensor.dtype().code == kDLInt && tensor.dtype().bits == 64;
}

auto is_cpu_bool_vector(const tvm::ffi::TensorView tensor) -> bool {
  return tensor.ndim() == 1 && tensor.is_contiguous() &&
         tensor.device().device_type == kDLCPU &&
         tensor.dtype().code == kDLBool && tensor.dtype().bits == 8;
}

auto validate_context_sliding_inputs(
    const tvm::ffi::TensorView visible_until,
    const tvm::ffi::TensorView raw_positions,
    const tvm::ffi::TensorView true_positions, int64_t query_start,
    int64_t query_count, int64_t sliding_window) -> void {
  host::RuntimeCheck(is_cpu_int32_vector(visible_until) &&
                         is_cpu_int32_vector(raw_positions) &&
                         is_cpu_int32_vector(true_positions),
                     "Context sliding inputs must be contiguous CPU int32 vectors");
  host::RuntimeCheck(raw_positions.size(0) == true_positions.size(0),
                     "Context raw and true positions must have equal lengths");
  host::RuntimeCheck(query_start >= 0 && query_count > 0 &&
                         query_start + query_count <= raw_positions.size(0),
                     "Context sliding query bounds are invalid");
  host::RuntimeCheck(sliding_window >= 0,
                     "Context sliding window must be non-negative");

  const auto *raw = static_cast<const int32_t *>(raw_positions.data_ptr());
  const auto *position =
      static_cast<const int32_t *>(true_positions.data_ptr());
  const auto *expiry = static_cast<const int32_t *>(visible_until.data_ptr());
  for (int64_t i = 0; i < raw_positions.size(0); ++i) {
    host::RuntimeCheck(raw[i] >= 0 && raw[i] < visible_until.size(0),
                       "Context raw position is outside visibility metadata");
    host::RuntimeCheck(expiry[raw[i]] > raw[i],
                       "A Context token expires before it is computed");
    if (i > 0) {
      host::RuntimeCheck(raw[i - 1] < raw[i],
                         "Context raw positions must be strictly increasing");
      host::RuntimeCheck(position[i - 1] < position[i],
                         "Context true positions must be strictly increasing");
    }
  }
}

auto count_context_sliding_keys(
    const tvm::ffi::TensorView visible_until,
    const tvm::ffi::TensorView raw_positions,
    const tvm::ffi::TensorView true_positions, int64_t query_start,
    int64_t query_count, int64_t sliding_window,
    const tvm::ffi::TensorView key_lengths) -> void {
  validate_context_sliding_inputs(visible_until, raw_positions, true_positions,
                                  query_start, query_count, sliding_window);
  host::RuntimeCheck(is_cpu_int32_vector(key_lengths) &&
                         key_lengths.size(0) == query_count,
                     "Context key lengths must be a CPU int32 query vector");

  const auto *raw = static_cast<const int32_t *>(raw_positions.data_ptr());
  const auto *position =
      static_cast<const int32_t *>(true_positions.data_ptr());
  const auto *expiry = static_cast<const int32_t *>(visible_until.data_ptr());
  auto *lengths = static_cast<int32_t *>(key_lengths.data_ptr());
  int64_t left = 0;
  for (int64_t local_query = 0; local_query < query_count; ++local_query) {
    const int64_t query = query_start + local_query;
    const int64_t threshold =
        static_cast<int64_t>(position[query]) - sliding_window;
    while (left < query && static_cast<int64_t>(position[left]) < threshold) {
      ++left;
    }
    int64_t count = 1;
    for (int64_t key = left; key < query; ++key) {
      if (expiry[raw[key]] > raw[query]) ++count;
    }
    host::RuntimeCheck(count <= std::numeric_limits<int32_t>::max(),
                       "Context sliding row exceeds int32 capacity");
    lengths[local_query] = static_cast<int32_t>(count);
  }
}

auto fill_context_sliding_keys(
    const tvm::ffi::TensorView visible_until,
    const tvm::ffi::TensorView raw_positions,
    const tvm::ffi::TensorView true_positions, int64_t query_start,
    int64_t query_count, int64_t sliding_window,
    const tvm::ffi::TensorView key_offsets,
    const tvm::ffi::TensorView key_positions) -> void {
  validate_context_sliding_inputs(visible_until, raw_positions, true_positions,
                                  query_start, query_count, sliding_window);
  host::RuntimeCheck(is_cpu_int32_vector(key_offsets) &&
                         key_offsets.size(0) == query_count + 1 &&
                         is_cpu_int32_vector(key_positions),
                     "Context sliding outputs must be contiguous CPU int32 vectors");
  const auto *offsets = static_cast<const int32_t *>(key_offsets.data_ptr());
  host::RuntimeCheck(offsets[0] == 0 && offsets[query_count] == key_positions.size(0),
                     "Context sliding offsets do not cover output keys");

  const auto *raw = static_cast<const int32_t *>(raw_positions.data_ptr());
  const auto *position =
      static_cast<const int32_t *>(true_positions.data_ptr());
  const auto *expiry = static_cast<const int32_t *>(visible_until.data_ptr());
  auto *output = static_cast<int32_t *>(key_positions.data_ptr());
  int64_t left = 0;
  for (int64_t local_query = 0; local_query < query_count; ++local_query) {
    const int64_t query = query_start + local_query;
    const int64_t threshold =
        static_cast<int64_t>(position[query]) - sliding_window;
    while (left < query && static_cast<int64_t>(position[left]) < threshold) {
      ++left;
    }
    int64_t cursor = offsets[local_query];
    for (int64_t key = left; key < query; ++key) {
      if (expiry[raw[key]] > raw[query]) output[cursor++] = key;
    }
    output[cursor++] = query;
    host::RuntimeCheck(cursor == offsets[local_query + 1],
                       "Context sliding count/fill passes disagree");
  }
}

auto validate_occurrence_sliding_inputs(
    const tvm::ffi::TensorView occurrence_positions,
    const tvm::ffi::TensorView query_starts,
    const tvm::ffi::TensorView query_ends,
    const tvm::ffi::TensorView key_offsets,
    const tvm::ffi::TensorView flat_keys,
    const tvm::ffi::TensorView true_positions, int64_t cached_len,
    int64_t device_len, int64_t sliding_window) -> void {
  host::RuntimeCheck(is_cpu_int32_vector(occurrence_positions) &&
                         is_cpu_int32_vector(query_starts) &&
                         is_cpu_int32_vector(query_ends) &&
                         is_cpu_int32_vector(key_offsets) &&
                         is_cpu_int32_vector(flat_keys) &&
                         is_cpu_int32_vector(true_positions),
                     "Occurrence sliding inputs must be contiguous CPU int32 vectors");
  host::RuntimeCheck(query_starts.size(0) == query_ends.size(0) &&
                         key_offsets.size(0) == query_starts.size(0) + 1,
                     "Occurrence segment metadata lengths disagree");
  host::RuntimeCheck(key_offsets.size(0) > 1 &&
                         key_offsets.size(0) <=
                             std::numeric_limits<int32_t>::max(),
                     "Occurrence segment metadata is invalid");
  host::RuntimeCheck(cached_len >= 0 && cached_len < device_len &&
                         device_len <= true_positions.size(0),
                     "Occurrence sliding query bounds are invalid");
  host::RuntimeCheck(sliding_window >= 0,
                     "Occurrence sliding window must be non-negative");
  const auto *offsets = static_cast<const int32_t *>(key_offsets.data_ptr());
  host::RuntimeCheck(offsets[0] == 0 &&
                         offsets[query_starts.size(0)] == flat_keys.size(0),
                     "Occurrence key offsets do not cover flat keys");
  for (int64_t segment = 0; segment < query_starts.size(0); ++segment) {
    host::RuntimeCheck(offsets[segment] >= 0 &&
                           offsets[segment] <= offsets[segment + 1],
                       "Occurrence key offsets must be monotonic");
  }
}

auto count_occurrence_sliding_keys(
    const tvm::ffi::TensorView occurrence_raw_tokens,
    const tvm::ffi::TensorView occurrence_positions,
    const tvm::ffi::TensorView query_starts,
    const tvm::ffi::TensorView query_ends,
    const tvm::ffi::TensorView key_offsets,
    const tvm::ffi::TensorView flat_keys,
    const tvm::ffi::TensorView true_positions, int64_t cached_len,
    int64_t device_len, int64_t initial_cached_len, int64_t sliding_window,
    const tvm::ffi::TensorView key_lengths,
    const tvm::ffi::TensorView cached_mask,
    const tvm::ffi::TensorView status) -> void {
  validate_occurrence_sliding_inputs(
      occurrence_positions, query_starts, query_ends, key_offsets, flat_keys,
      true_positions, cached_len, device_len, sliding_window);
  host::RuntimeCheck(is_cpu_int32_vector(occurrence_raw_tokens) &&
                         occurrence_raw_tokens.size(0) == occurrence_positions.size(0),
                     "Occurrence raw tokens must align with occurrence positions");
  const int64_t query_count = device_len - cached_len;
  host::RuntimeCheck(initial_cached_len >= 0 &&
                         initial_cached_len <= true_positions.size(0),
                     "Occurrence initial cached length is invalid");
  host::RuntimeCheck(is_cpu_int32_vector(key_lengths) &&
                         key_lengths.size(0) == query_count &&
                         is_cpu_bool_vector(cached_mask) &&
                         cached_mask.size(0) == initial_cached_len &&
                         is_cpu_int64_vector(status) && status.size(0) >= 1,
                     "Occurrence sliding count outputs have invalid layouts");

  const auto *raw =
      static_cast<const int32_t *>(occurrence_raw_tokens.data_ptr());
  const auto *positions =
      static_cast<const int32_t *>(occurrence_positions.data_ptr());
  const auto *starts = static_cast<const int32_t *>(query_starts.data_ptr());
  const auto *ends = static_cast<const int32_t *>(query_ends.data_ptr());
  const auto *offsets = static_cast<const int32_t *>(key_offsets.data_ptr());
  const auto *keys = static_cast<const int32_t *>(flat_keys.data_ptr());
  const auto *query_positions =
      static_cast<const int32_t *>(true_positions.data_ptr());
  auto *lengths = static_cast<int32_t *>(key_lengths.data_ptr());
  auto *used_cached = static_cast<bool *>(cached_mask.data_ptr());
  auto *result_status = static_cast<int64_t *>(status.data_ptr());
  result_status[0] = 0;
  int64_t local_query = 0;
  int64_t expected_query = cached_len;
  for (int64_t segment = 0; segment < query_starts.size(0); ++segment) {
    const int64_t raw_start = starts[segment];
    const int64_t raw_end = ends[segment];
    host::RuntimeCheck(raw_start >= 0 && raw_start < raw_end &&
                           raw_end <= true_positions.size(0),
                       "Occurrence segment bounds are invalid");
    const int64_t key_begin = offsets[segment];
    const int64_t key_end = offsets[segment + 1];
    const int64_t prefix_length = key_end - key_begin - (raw_end - raw_start);
    host::RuntimeCheck(prefix_length >= 0,
                       "Occurrence segment has fewer keys than local queries");
    for (int64_t cursor = key_begin; cursor < key_end; ++cursor) {
      host::RuntimeCheck(keys[cursor] >= 0 &&
                             keys[cursor] < occurrence_positions.size(0),
                         "Occurrence segment references an invalid occurrence");
      host::RuntimeCheck(raw[keys[cursor]] >= 0 &&
                             raw[keys[cursor]] < true_positions.size(0),
                         "Occurrence segment references an invalid raw token");
      if (cursor > key_begin && positions[keys[cursor - 1]] >= positions[keys[cursor]]) {
        result_status[0] = 1;
        return;
      }
    }

    const int64_t query_begin = std::max<int64_t>(raw_start, cached_len);
    const int64_t query_end = std::min<int64_t>(raw_end, device_len);
    if (query_begin >= query_end) continue;
    host::RuntimeCheck(query_begin == expected_query,
                       "Occurrence segments do not preserve flattened query order");
    for (int64_t query = query_begin; query < query_end; ++query) {
      const int64_t causal_count = prefix_length + (query - raw_start) + 1;
      const int64_t threshold =
          static_cast<int64_t>(query_positions[query]) - sliding_window;
      const auto *begin = keys + key_begin;
      const auto *causal_end = begin + causal_count;
      const auto *left = std::lower_bound(
          begin, causal_end, threshold,
          [positions](int32_t occurrence, int64_t value) {
            return static_cast<int64_t>(positions[occurrence]) < value;
          });
      const int64_t count = causal_end - left;
      host::RuntimeCheck(count > 0 && count <= std::numeric_limits<int32_t>::max(),
                         "Occurrence sliding row has an invalid key count");
      lengths[local_query++] = static_cast<int32_t>(count);
      for (const auto *selected = left; selected < causal_end; ++selected) {
        const int32_t selected_raw = raw[*selected];
        host::RuntimeCheck(selected_raw >= 0,
                           "Occurrence raw token must be non-negative");
        if (selected_raw < initial_cached_len) used_cached[selected_raw] = true;
      }
    }
    expected_query = query_end;
  }
  host::RuntimeCheck(expected_query == device_len && local_query == query_count,
                     "Occurrence segments do not cover the request extension");
}

auto fill_occurrence_sliding_keys(
    const tvm::ffi::TensorView occurrence_positions,
    const tvm::ffi::TensorView query_starts,
    const tvm::ffi::TensorView query_ends,
    const tvm::ffi::TensorView key_offsets,
    const tvm::ffi::TensorView flat_keys,
    const tvm::ffi::TensorView true_positions, int64_t cached_len,
    int64_t device_len, int64_t sliding_window, int64_t occurrence_base,
    const tvm::ffi::TensorView output_offsets,
    const tvm::ffi::TensorView key_positions) -> void {
  validate_occurrence_sliding_inputs(
      occurrence_positions, query_starts, query_ends, key_offsets, flat_keys,
      true_positions, cached_len, device_len, sliding_window);
  const int64_t query_count = device_len - cached_len;
  host::RuntimeCheck(occurrence_base >= 0 &&
                         occurrence_base + occurrence_positions.size(0) <=
                             std::numeric_limits<int32_t>::max(),
                     "Occurrence batch key positions exceed int32 capacity");
  host::RuntimeCheck(is_cpu_int32_vector(output_offsets) &&
                         output_offsets.size(0) == query_count + 1 &&
                         is_cpu_int32_vector(key_positions),
                     "Occurrence sliding outputs must be contiguous CPU int32 vectors");
  const auto *out_offsets =
      static_cast<const int32_t *>(output_offsets.data_ptr());
  host::RuntimeCheck(out_offsets[0] == 0 &&
                         out_offsets[query_count] == key_positions.size(0),
                     "Occurrence sliding offsets do not cover output keys");

  const auto *positions =
      static_cast<const int32_t *>(occurrence_positions.data_ptr());
  const auto *starts = static_cast<const int32_t *>(query_starts.data_ptr());
  const auto *ends = static_cast<const int32_t *>(query_ends.data_ptr());
  const auto *offsets = static_cast<const int32_t *>(key_offsets.data_ptr());
  const auto *keys = static_cast<const int32_t *>(flat_keys.data_ptr());
  const auto *query_positions =
      static_cast<const int32_t *>(true_positions.data_ptr());
  auto *output = static_cast<int32_t *>(key_positions.data_ptr());
  int64_t local_query = 0;
  for (int64_t segment = 0; segment < query_starts.size(0); ++segment) {
    const int64_t raw_start = starts[segment];
    const int64_t raw_end = ends[segment];
    const int64_t key_begin = offsets[segment];
    const int64_t key_end = offsets[segment + 1];
    const int64_t prefix_length = key_end - key_begin - (raw_end - raw_start);
    const int64_t query_begin = std::max<int64_t>(raw_start, cached_len);
    const int64_t query_end = std::min<int64_t>(raw_end, device_len);
    if (query_begin >= query_end) continue;
    for (int64_t query = query_begin; query < query_end; ++query) {
      const int64_t causal_count = prefix_length + (query - raw_start) + 1;
      const int64_t threshold =
          static_cast<int64_t>(query_positions[query]) - sliding_window;
      const auto *begin = keys + key_begin;
      const auto *causal_end = begin + causal_count;
      const auto *left = std::lower_bound(
          begin, causal_end, threshold,
          [positions](int32_t occurrence, int64_t value) {
            return static_cast<int64_t>(positions[occurrence]) < value;
          });
      int64_t cursor = out_offsets[local_query];
      for (const auto *selected = left; selected < causal_end; ++selected) {
        output[cursor++] = static_cast<int32_t>(occurrence_base + *selected);
      }
      host::RuntimeCheck(cursor == out_offsets[local_query + 1],
                         "Occurrence sliding count/fill passes disagree");
      ++local_query;
    }
  }
  host::RuntimeCheck(local_query == query_count,
                     "Occurrence sliding fill did not cover all queries");
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
TVM_FFI_DLL_EXPORT_TYPED_FUNC(count_context_sliding_keys,
                              count_context_sliding_keys);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fill_context_sliding_keys,
                              fill_context_sliding_keys);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(count_occurrence_sliding_keys,
                              count_occurrence_sliding_keys);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fill_occurrence_sliding_keys,
                              fill_occurrence_sliding_keys);
