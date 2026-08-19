#include <minisgl/utils.h>

#include <cstdint>
#include <queue>
#include <unordered_map>
#include <vector>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

namespace {

auto is_cpu_int_tensor(const tvm::ffi::TensorView tensor, int ndim,
                       int bits) -> bool {
  return tensor.ndim() == ndim && tensor.is_contiguous() &&
         tensor.device().device_type == kDLCPU &&
         tensor.dtype().code == kDLInt && tensor.dtype().bits == bits;
}

struct Node {
  std::unordered_map<int32_t, int32_t> next;
  int32_t failure = 0;
  int32_t output_link = -1;
  std::vector<int32_t> terminal_patterns;
};

auto transition(const std::vector<Node> &trie, int32_t node,
                int32_t byte) -> int32_t {
  const auto found = trie[node].next.find(byte);
  return found == trie[node].next.end() ? -1 : found->second;
}

auto aho_find_all(const tvm::ffi::TensorView source,
                  const tvm::ffi::TensorView flat_patterns,
                  const tvm::ffi::TensorView pattern_offsets,
                  const tvm::ffi::TensorView matches,
                  int64_t max_matches) -> int64_t {
  host::RuntimeCheck(is_cpu_int_tensor(source, 1, 32),
                     "source must be a contiguous 1D CPU int32 tensor");
  host::RuntimeCheck(is_cpu_int_tensor(flat_patterns, 1, 32),
                     "flat_patterns must be a contiguous 1D CPU int32 tensor");
  host::RuntimeCheck(is_cpu_int_tensor(pattern_offsets, 1, 64),
                     "pattern_offsets must be a contiguous 1D CPU int64 tensor");
  host::RuntimeCheck(is_cpu_int_tensor(matches, 2, 64) && matches.size(1) == 3,
                     "matches must be a contiguous [N, 3] CPU int64 tensor");
  host::RuntimeCheck(max_matches > 0 && matches.size(0) >= max_matches,
                     "matches capacity is smaller than max_matches");
  host::RuntimeCheck(pattern_offsets.size(0) >= 1,
                     "pattern_offsets must contain at least zero");

  const auto *text = static_cast<const int32_t *>(source.data_ptr());
  const auto *patterns = static_cast<const int32_t *>(flat_patterns.data_ptr());
  const auto *offsets = static_cast<const int64_t *>(pattern_offsets.data_ptr());
  auto *result = static_cast<int64_t *>(matches.data_ptr());
  const int64_t pattern_count = pattern_offsets.size(0) - 1;

  std::vector<Node> trie(1);
  std::vector<int64_t> lengths(pattern_count);
  for (int64_t pattern_id = 0; pattern_id < pattern_count; ++pattern_id) {
    const int64_t begin = offsets[pattern_id];
    const int64_t end = offsets[pattern_id + 1];
    host::RuntimeCheck(begin >= 0 && end > begin &&
                           end <= flat_patterns.size(0),
                       "pattern_offsets contains an invalid or empty pattern");
    lengths[pattern_id] = end - begin;
    int32_t node = 0;
    for (int64_t pos = begin; pos < end; ++pos) {
      const int32_t byte = patterns[pos];
      host::RuntimeCheck(byte >= 0 && byte <= 255,
                         "patterns must contain byte values in [0, 255]");
      const auto found = trie[node].next.find(byte);
      if (found == trie[node].next.end()) {
        const auto child = static_cast<int32_t>(trie.size());
        trie[node].next.emplace(byte, child);
        trie.emplace_back();
        node = child;
      } else {
        node = found->second;
      }
    }
    trie[node].terminal_patterns.push_back(static_cast<int32_t>(pattern_id));
  }

  std::queue<int32_t> queue;
  for (const auto &[byte, child] : trie[0].next) {
    (void)byte;
    trie[child].failure = 0;
    queue.push(child);
  }
  while (!queue.empty()) {
    const int32_t node = queue.front();
    queue.pop();
    for (const auto &[byte, child] : trie[node].next) {
      int32_t fallback = trie[node].failure;
      int32_t fallback_child = transition(trie, fallback, byte);
      while (fallback != 0 && fallback_child == -1) {
        fallback = trie[fallback].failure;
        fallback_child = transition(trie, fallback, byte);
      }
      if (fallback_child != -1) {
        fallback = fallback_child;
      }
      trie[child].failure = fallback;
      trie[child].output_link = !trie[fallback].terminal_patterns.empty()
                                    ? fallback
                                    : trie[fallback].output_link;
      queue.push(child);
    }
  }

  int64_t count = 0;
  int32_t node = 0;
  for (int64_t pos = 0; pos < source.size(0); ++pos) {
    const int32_t byte = text[pos];
    host::RuntimeCheck(byte >= 0 && byte <= 255,
                       "source must contain byte values in [0, 255]");
    int32_t child = transition(trie, node, byte);
    while (node != 0 && child == -1) {
      node = trie[node].failure;
      child = transition(trie, node, byte);
    }
    if (child != -1) {
      node = child;
    }
    for (int32_t output_node = node; output_node != -1;
         output_node = trie[output_node].output_link) {
      for (const int32_t pattern_id : trie[output_node].terminal_patterns) {
        if (count >= max_matches) {
          return -1;
        }
        result[count * 3] = pattern_id;
        result[count * 3 + 1] = pos + 1 - lengths[pattern_id];
        result[count * 3 + 2] = pos + 1;
        ++count;
      }
    }
  }
  return count;
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(aho_find_all, aho_find_all);
