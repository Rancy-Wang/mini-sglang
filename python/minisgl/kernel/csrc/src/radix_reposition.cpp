#include <minisgl/utils.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

namespace {

constexpr int32_t kToken = 0;
constexpr int32_t kDelta = 1;
constexpr int32_t kReposition = 2;

auto is_cpu_tensor(const tvm::ffi::TensorView tensor, int ndim, int bits,
                   int code = kDLInt) -> bool {
  return tensor.ndim() == ndim && tensor.is_contiguous() &&
         tensor.device().device_type == kDLCPU &&
         tensor.dtype().code == code && tensor.dtype().bits == bits;
}

auto compile_radix_reposition_layout(
    const tvm::ffi::TensorView token_ids,
    const tvm::ffi::TensorView drop_insert_offsets,
    const tvm::ffi::TensorView drop_range_offsets,
    const tvm::ffi::TensorView drop_ranges,
    const tvm::ffi::TensorView delta_marker_ids,
    const tvm::ffi::TensorView reposition_raw_boundaries,
    const tvm::ffi::TensorView reposition_insert_offsets,
    const tvm::ffi::TensorView records,
    const tvm::ffi::TensorView virtual_mask,
    const tvm::ffi::TensorView key_to_token,
    const tvm::ffi::TensorView token_to_key,
    const tvm::ffi::TensorView positions,
    const tvm::ffi::TensorView repos_info,
    const tvm::ffi::TensorView keep_mask,
    const tvm::ffi::TensorView materialized_stage,
    const tvm::ffi::TensorView effective_repositions,
    const tvm::ffi::TensorView ignored_repositions,
    const tvm::ffi::TensorView status) -> void {
  host::RuntimeCheck(is_cpu_tensor(token_ids, 1, 32) ||
                         is_cpu_tensor(token_ids, 1, 64),
                     "token_ids must be a CPU int32/int64 vector");
  host::RuntimeCheck(is_cpu_tensor(drop_insert_offsets, 1, 32) &&
                         is_cpu_tensor(drop_range_offsets, 1, 32) &&
                         is_cpu_tensor(drop_ranges, 1, 32) &&
                         is_cpu_tensor(delta_marker_ids, 1, 32),
                     "Drop inputs must be contiguous CPU int32 vectors");
  host::RuntimeCheck(is_cpu_tensor(reposition_raw_boundaries, 1, 32) &&
                         is_cpu_tensor(reposition_insert_offsets, 1, 32),
                     "Reposition inputs must be contiguous CPU int32 vectors");
  host::RuntimeCheck(is_cpu_tensor(records, 2, 32) && records.size(1) == 4,
                     "records must be a contiguous CPU int32 [N, 4] tensor");
  host::RuntimeCheck(is_cpu_tensor(virtual_mask, 1, 8, kDLBool) &&
                         is_cpu_tensor(keep_mask, 1, 8, kDLBool) &&
                         is_cpu_tensor(effective_repositions, 1, 8, kDLBool) &&
                         is_cpu_tensor(ignored_repositions, 1, 8, kDLBool),
                     "mask outputs must be contiguous CPU bool vectors");
  host::RuntimeCheck(is_cpu_tensor(key_to_token, 1, 64) &&
                         is_cpu_tensor(token_to_key, 1, 64) &&
                         is_cpu_tensor(status, 1, 64),
                     "mapping/status outputs must be contiguous CPU int64 vectors");
  host::RuntimeCheck(is_cpu_tensor(positions, 1, 32) &&
                         is_cpu_tensor(repos_info, 1, 32) &&
                         is_cpu_tensor(materialized_stage, 1, 32),
                     "token metadata outputs must be contiguous CPU int32 vectors");

  const int64_t token_count = token_ids.size(0);
  const int64_t drop_count = drop_insert_offsets.size(0);
  const int64_t reposition_count = reposition_raw_boundaries.size(0);
  host::RuntimeCheck(token_count <= std::numeric_limits<int32_t>::max(),
                     "The token stream exceeds the int32 Reposition limit");
  host::RuntimeCheck(drop_range_offsets.size(0) == drop_count + 1,
                     "drop_range_offsets must have E + 1 entries");
  host::RuntimeCheck(delta_marker_ids.size(0) == drop_count,
                     "delta_marker_ids must have one entry per Drop event");
  host::RuntimeCheck(drop_ranges.size(0) % 2 == 0,
                     "drop_ranges must contain start/end pairs");
  host::RuntimeCheck(reposition_insert_offsets.size(0) == reposition_count,
                     "Reposition boundaries and offsets must align");
  host::RuntimeCheck(records.size(0) >= token_count + drop_count + reposition_count &&
                         virtual_mask.size(0) >= records.size(0) &&
                         key_to_token.size(0) >= records.size(0),
                     "Radix output capacity is too small");
  host::RuntimeCheck(token_to_key.size(0) == token_count &&
                         positions.size(0) == token_count &&
                         repos_info.size(0) == token_count &&
                         keep_mask.size(0) == token_count &&
                         materialized_stage.size(0) == token_count,
                     "Per-token output lengths are inconsistent");
  host::RuntimeCheck(effective_repositions.size(0) == reposition_count &&
                         ignored_repositions.size(0) == reposition_count &&
                         status.size(0) >= 6,
                     "Reposition output lengths are inconsistent");

  const auto *drops = static_cast<const int32_t *>(drop_insert_offsets.data_ptr());
  const auto *drop_offsets = static_cast<const int32_t *>(drop_range_offsets.data_ptr());
  const auto *ranges = static_cast<const int32_t *>(drop_ranges.data_ptr());
  const auto *markers = static_cast<const int32_t *>(delta_marker_ids.data_ptr());
  const auto *raw_boundaries =
      static_cast<const int32_t *>(reposition_raw_boundaries.data_ptr());
  const auto *reposition_offsets =
      static_cast<const int32_t *>(reposition_insert_offsets.data_ptr());
  const int64_t range_count = drop_ranges.size(0) / 2;

  host::RuntimeCheck(drop_offsets[0] == 0 && drop_offsets[drop_count] == range_count,
                     "drop_range_offsets does not cover drop_ranges");
  for (int64_t i = 0; i < drop_count; ++i) {
    host::RuntimeCheck(drops[i] >= 0 && drops[i] <= token_count,
                       "Drop insertion offset is outside the token stream");
    host::RuntimeCheck(i == 0 || drops[i - 1] < drops[i],
                       "Drop insertion offsets must be strictly increasing");
    host::RuntimeCheck(drop_offsets[i] < drop_offsets[i + 1],
                       "Every Drop event must own a non-empty range set");
  }
  for (int64_t i = 0; i < reposition_count; ++i) {
    host::RuntimeCheck(reposition_offsets[i] == raw_boundaries[i] + 1 &&
                           reposition_offsets[i] > 0 &&
                           reposition_offsets[i] <= token_count,
                       "Reposition raw boundary and insertion offset disagree");
    host::RuntimeCheck(i == 0 || raw_boundaries[i - 1] < raw_boundaries[i],
                       "Reposition raw boundaries must be strictly increasing");
  }

  auto *position = static_cast<int32_t *>(positions.data_ptr());
  auto *repos = static_cast<int32_t *>(repos_info.data_ptr());
  auto *kept = static_cast<bool *>(keep_mask.data_ptr());
  auto *ready = static_cast<int32_t *>(materialized_stage.data_ptr());
  auto *effective = static_cast<bool *>(effective_repositions.data_ptr());
  auto *ignored = static_cast<bool *>(ignored_repositions.data_ptr());
  auto *result_status = static_cast<int64_t *>(status.data_ptr());
  std::fill(effective, effective + reposition_count, false);
  std::fill(ignored, ignored + reposition_count, false);

  std::vector<int32_t> previous(token_count, -1);
  std::vector<int32_t> next(token_count, -1);
  int32_t head = -1;
  int32_t tail = -1;
  int32_t first_noncompact = -1;
  int32_t active_count = 0;
  int32_t next_position = 0;
  int32_t current_reposition = -1;
  int32_t stage = 0;
  int64_t drop_idx = 0;
  int64_t reposition_idx = 0;

  for (int32_t insertion = 0; insertion <= token_count; ++insertion) {
    if (drop_idx < drop_count && drops[drop_idx] == insertion) {
      int32_t previous_end = -1;
      for (int32_t range_idx = drop_offsets[drop_idx];
           range_idx < drop_offsets[drop_idx + 1]; ++range_idx) {
        const int32_t start = ranges[2 * range_idx];
        const int32_t end = ranges[2 * range_idx + 1];
        host::RuntimeCheck(start >= 0 && start < end && end <= insertion &&
                               (range_idx == drop_offsets[drop_idx] || previous_end < start),
                           "Drop ranges must be ordered, disjoint, and already materialized");
        previous_end = end;
        for (int32_t token = start; token < end; ++token) {
          if (!kept[token]) {
            continue;
          }
          const int32_t before = previous[token];
          const int32_t after = next[token];
          if (before < 0) {
            head = after;
          } else {
            next[before] = after;
          }
          if (after < 0) {
            tail = before;
          } else {
            previous[after] = before;
          }
          kept[token] = false;
          --active_count;
          if (first_noncompact == token) {
            first_noncompact = after;
          }
          if (after >= 0 &&
              (first_noncompact < 0 || after < first_noncompact)) {
            first_noncompact = after;
          }
        }
      }
      if (active_count == 0) {
        first_noncompact = -1;
      }
      ++drop_idx;
    }

    if (reposition_idx < reposition_count &&
        reposition_offsets[reposition_idx] == insertion) {
      if (active_count == 0) {
        result_status[0] = 1;
        result_status[4] = raw_boundaries[reposition_idx];
        return;
      }
      if (first_noncompact < 0) {
        ignored[reposition_idx] = true;
      } else {
        ++stage;
        effective[reposition_idx] = true;
        int32_t rank = previous[first_noncompact] < 0
                           ? 0
                           : position[previous[first_noncompact]] + 1;
        for (int32_t token = first_noncompact; token >= 0; token = next[token]) {
          host::RuntimeCheck(position[token] > rank,
                             "Non-compact active suffix invariant was violated");
          position[token] = rank++;
          repos[token] = raw_boundaries[reposition_idx];
          ready[token] = stage;
        }
        current_reposition = raw_boundaries[reposition_idx];
        next_position = active_count;
        first_noncompact = -1;
      }
      ++reposition_idx;
    }

    if (insertion == token_count) {
      continue;
    }
    const int32_t token = insertion;
    const int32_t rank = active_count;
    kept[token] = true;
    position[token] = next_position++;
    repos[token] = current_reposition;
    ready[token] = stage;
    previous[token] = tail;
    if (tail < 0) {
      head = token;
    } else {
      next[tail] = token;
    }
    tail = token;
    ++active_count;
    if (first_noncompact < 0 && position[token] != rank) {
      first_noncompact = token;
    }
  }

  auto *output = static_cast<int32_t *>(records.data_ptr());
  auto *is_virtual = static_cast<bool *>(virtual_mask.data_ptr());
  auto *key_token = static_cast<int64_t *>(key_to_token.data_ptr());
  auto *token_key = static_cast<int64_t *>(token_to_key.data_ptr());
  const auto *tokens32 = token_ids.dtype().bits == 32
                             ? static_cast<const int32_t *>(token_ids.data_ptr())
                             : nullptr;
  const auto *tokens64 = token_ids.dtype().bits == 64
                             ? static_cast<const int64_t *>(token_ids.data_ptr())
                             : nullptr;
  int64_t key_idx = 0;
  drop_idx = 0;
  reposition_idx = 0;
  for (int32_t insertion = 0; insertion <= token_count; ++insertion) {
    if (drop_idx < drop_count && drops[drop_idx] == insertion) {
      output[key_idx * 4] = kDelta;
      output[key_idx * 4 + 1] = markers[drop_idx];
      output[key_idx * 4 + 2] = -1;
      output[key_idx * 4 + 3] = -1;
      is_virtual[key_idx] = true;
      key_token[key_idx++] = -1;
      ++drop_idx;
    }
    if (reposition_idx < reposition_count &&
        reposition_offsets[reposition_idx] == insertion) {
      if (effective[reposition_idx]) {
        output[key_idx * 4] = kReposition;
        output[key_idx * 4 + 1] = raw_boundaries[reposition_idx];
        output[key_idx * 4 + 2] = -1;
        output[key_idx * 4 + 3] = -1;
        is_virtual[key_idx] = true;
        key_token[key_idx++] = -1;
      }
      ++reposition_idx;
    }
    if (insertion == token_count) {
      continue;
    }
    token_key[insertion] = key_idx;
    const int64_t token_id = tokens32 == nullptr ? tokens64[insertion]
                                                  : tokens32[insertion];
    if (token_id < 0 || token_id > std::numeric_limits<int32_t>::max()) {
      result_status[0] = 2;
      result_status[4] = token_id;
      return;
    }
    output[key_idx * 4] = kToken;
    output[key_idx * 4 + 1] = static_cast<int32_t>(token_id);
    output[key_idx * 4 + 2] = repos[insertion];
    output[key_idx * 4 + 3] = position[insertion];
    is_virtual[key_idx] = false;
    key_token[key_idx++] = insertion;
  }

  result_status[0] = 0;
  result_status[1] = key_idx;
  result_status[2] = stage;
  result_status[3] = next_position;
  result_status[4] = -1;
  result_status[5] = current_reposition;
  (void)head;
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(compile_radix_reposition_layout,
                              compile_radix_reposition_layout);
