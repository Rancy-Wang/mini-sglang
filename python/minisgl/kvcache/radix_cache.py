from __future__ import annotations

import heapq
import time
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


def _edge_key(key_fn: KEY_FN, key: torch.Tensor, virtual_mask: torch.Tensor) -> tuple[Any, bool]:
    return key_fn(key), bool(virtual_mask[0].item())


class RadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        # Drop-aware eviction keeps structural references and physical KV users separate.
        self.kv_need_leaf_count: int = 0
        self.kv_pin_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._virtual_mask: torch.Tensor
        self._length: int
        self._page_length: int
        self._resident: bool

    def set_key_value(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        virtual_mask: torch.Tensor | None = None,
    ) -> None:
        if virtual_mask is None:
            virtual_mask = torch.zeros(len(key), dtype=torch.bool, device="cpu")
        assert len(key) == len(value) == len(virtual_mask)
        self._key = key
        self._value = value
        self._virtual_mask = virtual_mask
        self._length = len(key)
        self._page_length = int(torch.count_nonzero(~virtual_mask).item())
        real_value_mask = (~virtual_mask).to(device=value.device, non_blocking=True)
        real_values = value[real_value_mask]
        if len(real_values) == 0:
            self._resident = False
        else:
            all_resident = bool(torch.all(real_values >= 0).item())
            self._resident = all_resident

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[_edge_key(self.key_fn, self._key, self._virtual_mask)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def page_length(self) -> int:
        return self._page_length

    @property
    def resident(self) -> bool:
        return self._resident

    @property
    def resident_page_length(self) -> int:
        return self._page_length if self._resident else 0

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    @property
    def virtual_mask(self) -> torch.Tensor:
        return self._virtual_mask

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor, virtual_mask: torch.Tensor) -> int:
        from minisgl.kernel.radix import fast_compare_radix_key

        return fast_compare_radix_key(self._key, input_ids, self._virtual_mask, virtual_mask)

    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos], self._virtual_mask[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count
        new_node.kv_need_leaf_count = self.kv_need_leaf_count

        self.set_key_value(self._key[pos:], self._value[pos:], self._virtual_mask[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode
    pinned_slots: tuple[int, ...] | None = None

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list)

    def get_matched_virtual_mask(self) -> torch.Tensor:
        node = self.node
        mask_list: List[torch.Tensor] = []
        while not node.is_root():
            mask_list.append(node.virtual_mask)
            node = node.parent
        mask_list.reverse()
        return torch.cat(mask_list)

    @property
    def physical_cached_len(self) -> int:
        node = self.node
        length = 0
        while not node.is_root():
            length += node.resident_page_length
            node = node.parent
        return length

    def with_pinned_slots(self, slots: torch.Tensor) -> RadixCacheHandle:
        return replace(self, pinned_slots=tuple(int(slot) for slot in slots.tolist()))


@dataclass(frozen=True)
class DropAwareInsertResult:
    existing_prefix_len: int
    handle: RadixCacheHandle
    canonical_indices: torch.Tensor


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected
        self.delta_marker_registry = None
        self.drop_aware_eviction = False
        self._slot_owner: Dict[int, RadixTreeNode] = {}
        self._slot_pin_count: Counter[int] = Counter()

    def enable_drop_aware_eviction(self) -> None:
        if self.root_node.children:
            raise RuntimeError("Drop-aware eviction must be enabled before Radix insertion.")
        if self.page_size != 1:
            raise ValueError("Drop-aware eviction requires page_size=1.")
        self.drop_aware_eviction = True

    def bind_delta_marker_registry(self, registry) -> None:
        if self.root_node.children:
            raise RuntimeError("Delta marker registry must be bound before Radix insertion.")
        self.delta_marker_registry = registry

    @staticmethod
    def _marker_ids(key: torch.Tensor, virtual_mask: torch.Tensor) -> List[int]:
        if not bool(torch.any(virtual_mask).item()):
            return []
        return [int(value) for value in key[virtual_mask].tolist()]

    @staticmethod
    def _path_nodes(node: RadixTreeNode) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = []
        while not node.is_root():
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def with_pinned_slots(
        self, handle: BaseCacheHandle, slots: torch.Tensor
    ) -> RadixCacheHandle:
        assert isinstance(handle, RadixCacheHandle)
        return handle.with_pinned_slots(slots)

    def _is_drop_reclaimable(self, node: RadixTreeNode) -> bool:
        return (
            self.drop_aware_eviction
            and node.resident
            and node.kv_need_leaf_count == 0
            and node.kv_pin_count == 0
        )

    def _is_evictable(self, node: RadixTreeNode) -> bool:
        return node.resident and (node.ref_count == 0 or self._is_drop_reclaimable(node))

    def _remove_size(self, node: RadixTreeNode) -> None:
        size = node.resident_page_length
        if size == 0:
            return
        if self._is_evictable(node):
            self.evictable_size -= size
        else:
            self.protected_size -= size

    def _add_size(self, node: RadixTreeNode) -> None:
        size = node.resident_page_length
        if size == 0:
            return
        if self._is_evictable(node):
            self.evictable_size += size
        else:
            self.protected_size += size

    def _real_slots(self, node: RadixTreeNode) -> List[int]:
        if not node.resident:
            return []
        real_mask = (~node.virtual_mask).to(device=node.value.device, non_blocking=True)
        return [int(slot) for slot in node.value[real_mask].tolist()]

    def _register_node_slots(self, node: RadixTreeNode) -> None:
        for slot in self._real_slots(node):
            previous = self._slot_owner.get(slot)
            if previous is not None and previous is not node:
                raise RuntimeError(f"KV slot {slot} is owned by multiple Radix nodes.")
            self._slot_owner[slot] = node

    def _unregister_node_slots(self, node: RadixTreeNode) -> None:
        for slot in self._real_slots(node):
            if self._slot_pin_count[slot] != 0:
                raise RuntimeError(f"Cannot release pinned KV slot {slot}.")
            if self._slot_owner.pop(slot, None) is not node:
                raise RuntimeError(f"Radix owner mismatch while releasing KV slot {slot}.")

    def _split_node(self, node: RadixTreeNode, pos: int) -> RadixTreeNode:
        self._remove_size(node)
        old_need_count = node.kv_need_leaf_count
        new_node = node.split_at(pos)
        new_node.kv_need_leaf_count = old_need_count
        new_node.kv_pin_count = sum(
            self._slot_pin_count[slot] for slot in self._real_slots(new_node)
        )
        node.kv_pin_count = sum(self._slot_pin_count[slot] for slot in self._real_slots(node))
        for slot in self._real_slots(new_node):
            previous = self._slot_owner.get(slot)
            if previous is not node:
                raise RuntimeError(f"Radix owner mismatch while splitting KV slot {slot}.")
            self._slot_owner[slot] = new_node
        self._register_node_slots(node)
        self._add_size(new_node)
        self._add_size(node)
        return new_node

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        if self.drop_aware_eviction:
            path_nodes = self._path_nodes(handle.node)
            if handle.pinned_slots is None:
                pinned_slots = tuple(
                    slot for path_node in path_nodes for slot in self._real_slots(path_node)
                )
            else:
                pinned_slots = handle.pinned_slots

            if unlock:
                for slot in pinned_slots:
                    owner = self._slot_owner.get(slot)
                    if owner is None:
                        raise RuntimeError(f"Pinned KV slot {slot} lost its Radix owner.")
                    self._remove_size(owner)
                    if self._slot_pin_count[slot] <= 0 or owner.kv_pin_count <= 0:
                        raise RuntimeError(f"KV slot {slot} pin-count underflow.")
                    self._slot_pin_count[slot] -= 1
                    owner.kv_pin_count -= 1
                    self._add_size(owner)
                for path_node in reversed(path_nodes):
                    self._remove_size(path_node)
                    path_node.ref_count -= 1
                    if path_node.ref_count < 0:
                        raise RuntimeError("Radix structural ref-count underflow.")
                    self._add_size(path_node)
            else:
                for path_node in path_nodes:
                    self._remove_size(path_node)
                    path_node.ref_count += 1
                    self._add_size(path_node)
                for slot in pinned_slots:
                    owner = self._slot_owner.get(slot)
                    if owner is None:
                        raise RuntimeError(f"Cannot pin unknown KV slot {slot}.")
                    self._remove_size(owner)
                    self._slot_pin_count[slot] += 1
                    owner.kv_pin_count += 1
                    self._add_size(owner)
            return

        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.page_length
                    self.protected_size -= node.page_length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.page_length
                    self.protected_size += node.page_length
                node.ref_count += 1
                node = node.parent

    def match_prefix(
        self, input_ids: torch.Tensor, virtual_mask: torch.Tensor | None = None
    ) -> MatchResult:
        virtual_mask = self._normalize_virtual_mask(input_ids, virtual_mask)
        node, prefix_len = self._tree_walk(input_ids, virtual_mask)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def _leaf_keep_mask(self, leaf: RadixTreeNode) -> torch.Tensor:
        path_nodes = self._path_nodes(leaf)
        token_count = sum(node.page_length for node in path_nodes)
        keep_mask = torch.ones(token_count, dtype=torch.bool, device="cpu")
        if self.delta_marker_registry is None:
            return keep_mask
        for path_node in path_nodes:
            for marker in self._marker_ids(path_node._key, path_node.virtual_mask):
                for start, end in self.delta_marker_registry.canonical_for(marker):
                    if start >= token_count:
                        continue
                    keep_mask[start : min(end, token_count)] = False
        return keep_mask

    def _adjust_leaf_contribution(
        self,
        leaf: RadixTreeNode,
        delta: int,
        keep_mask: torch.Tensor | None = None,
    ) -> None:
        if leaf.is_root():
            return
        path_nodes = self._path_nodes(leaf)
        if keep_mask is None:
            keep_mask = self._leaf_keep_mask(leaf)
        keep_mask = keep_mask.to(device="cpu", dtype=torch.bool)
        token_cursor = 0
        for path_node in path_nodes:
            has_virtual = bool(torch.any(path_node.virtual_mask).item())
            has_real = path_node.page_length > 0
            if has_virtual and has_real:
                raise RuntimeError("Drop-aware Radix nodes must not mix marker and token keys.")
            if not has_real:
                continue
            segment = keep_mask[token_cursor : token_cursor + path_node.page_length]
            if len(segment) != path_node.page_length:
                raise RuntimeError("Leaf keep mask is shorter than its Radix token path.")
            all_kept = bool(torch.all(segment).item())
            all_dropped = bool(torch.all(~segment).item())
            if not (all_kept or all_dropped):
                raise RuntimeError("A Drop boundary cuts through a normalized Radix node.")
            if all_kept:
                self._remove_size(path_node)
                path_node.kv_need_leaf_count += delta
                if path_node.kv_need_leaf_count < 0:
                    raise RuntimeError("Radix KV-need leaf count underflow.")
                self._add_size(path_node)
            token_cursor += path_node.page_length
        if token_cursor > len(keep_mask):
            raise RuntimeError("Leaf token path exceeds its keep mask.")

    @staticmethod
    def _segment_ends(
        indices: torch.Tensor,
        virtual_mask: torch.Tensor,
        key_to_token: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> List[int]:
        key_count = len(indices)
        if key_count == 0:
            return []
        real_mask = ~virtual_mask
        keep_by_key = torch.zeros(key_count, dtype=torch.bool, device="cpu")
        keep_by_key[real_mask] = keep_mask[key_to_token[real_mask]].to(dtype=torch.bool)
        resident_by_key = (indices >= 0).to(device="cpu", dtype=torch.bool)
        boundaries = torch.zeros(key_count + 1, dtype=torch.bool, device="cpu")
        boundaries[-1] = True
        if key_count > 1:
            adjacent_real = real_mask[:-1] & real_mask[1:]
            boundaries[1:key_count] = (
                virtual_mask[:-1]
                | virtual_mask[1:]
                | (
                    adjacent_real
                    & (
                        (keep_by_key[:-1] != keep_by_key[1:])
                        | (resident_by_key[:-1] != resident_by_key[1:])
                    )
                )
            )
        return torch.nonzero(boundaries, as_tuple=False).view(-1).tolist()

    def _merge_node_values(self, node: RadixTreeNode, candidates: torch.Tensor) -> None:
        if node.page_length == 0 or node.resident or bool(torch.all(candidates < 0).item()):
            return
        if bool(torch.any(candidates < 0).item()):
            raise RuntimeError("Cannot partially refill a Drop-aware Radix node.")
        self._remove_size(node)
        node.set_key_value(node._key, candidates.clone(), node.virtual_mask)
        self._register_node_slots(node)
        self._add_size(node)

    def _append_segment_chain(
        self,
        parent: RadixTreeNode,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
        virtual_mask: torch.Tensor,
        segment_ends: List[int],
        cursor: int,
    ) -> RadixTreeNode:
        for segment_end in segment_ends:
            if segment_end <= cursor:
                continue
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(
                input_ids[cursor:segment_end],
                indices[cursor:segment_end].clone(),
                virtual_mask[cursor:segment_end].clone(),
            )
            new_node.set_parent(parent)
            self._register_node_slots(new_node)
            self._add_size(new_node)
            if self.delta_marker_registry is not None:
                self.delta_marker_registry.add_tree_refs(
                    self._marker_ids(new_node._key, new_node.virtual_mask)
                )
            parent = new_node
            cursor = segment_end
        return parent

    def commit_drop_prefix(
        self,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
        virtual_mask: torch.Tensor,
        key_to_token: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> DropAwareInsertResult:
        if not self.drop_aware_eviction:
            raise RuntimeError("Drop-aware prefix commit is not enabled.")
        if self.delta_marker_registry is None and bool(torch.any(virtual_mask).item()):
            raise RuntimeError("Drop-aware marker commit requires a marker registry.")
        if not len(input_ids) == len(indices) == len(virtual_mask) == len(key_to_token):
            raise ValueError("Drop-aware Radix key-axis tensors must have equal lengths.")
        if len(input_ids) == 0:
            handle = RadixCacheHandle(0, self.root_node)
            return DropAwareInsertResult(0, handle, indices.clone())
        if bool(torch.any(indices[virtual_mask.to(indices.device)] != -1).item()):
            raise ValueError("Virtual Radix keys must use page value -1.")
        real_positions = key_to_token[~virtual_mask]
        if len(real_positions) > 0:
            if bool(torch.any(real_positions < 0).item()) or bool(
                torch.any(real_positions >= len(keep_mask)).item()
            ):
                raise ValueError("Drop-aware key-to-token mapping is outside the keep mask.")
            kept_key_mask = keep_mask[real_positions].to(
                device=indices.device, dtype=torch.bool, non_blocking=True
            )
            real_candidates = indices[(~virtual_mask).to(indices.device)]
            if bool(torch.any(real_candidates[kept_key_mask] < 0).item()):
                raise RuntimeError("A kept token reached Radix commit without a KV slot.")

        segment_ends = self._segment_ends(indices, virtual_mask, key_to_token, keep_mask)
        cursor = 0
        segment_idx = 0
        existing_prefix_len = len(input_ids)
        node = self.root_node
        added_leaf = False
        while cursor < len(input_ids):
            while segment_ends[segment_idx] <= cursor:
                segment_idx += 1
            desired_end = segment_ends[segment_idx]
            child = node.children.get(
                _edge_key(self.key_fn, input_ids[cursor:], virtual_mask[cursor:])
            )
            if child is None:
                existing_prefix_len = cursor
                if node.is_leaf() and not node.is_root():
                    self._adjust_leaf_contribution(node, -1)
                node = self._append_segment_chain(
                    node,
                    input_ids,
                    indices,
                    virtual_mask,
                    segment_ends,
                    cursor,
                )
                added_leaf = True
                break

            raw_match_len = child.get_match_len(
                input_ids[cursor:], virtual_mask[cursor:]
            )
            if raw_match_len <= 0:
                raise RuntimeError("Radix child edge matched without a key prefix.")
            allowed_len = desired_end - cursor
            match_len = min(raw_match_len, allowed_len)
            if match_len < child.length:
                prefix_node = self._split_node(child, match_len)
                self._merge_node_values(
                    prefix_node, indices[cursor : cursor + match_len]
                )
                node = prefix_node
                cursor += match_len
                suffix_matches = cursor < len(input_ids) and _edge_key(
                    self.key_fn,
                    input_ids[cursor:],
                    virtual_mask[cursor:],
                ) == _edge_key(self.key_fn, child._key, child.virtual_mask)
                if cursor < len(input_ids) and not suffix_matches:
                    existing_prefix_len = cursor
                    node = self._append_segment_chain(
                        node,
                        input_ids,
                        indices,
                        virtual_mask,
                        segment_ends,
                        cursor,
                    )
                    added_leaf = True
                    break
                continue

            self._merge_node_values(child, indices[cursor : cursor + child.length])
            node = child
            cursor += child.length

        if added_leaf:
            self._adjust_leaf_contribution(node, 1, keep_mask)
        handle = RadixCacheHandle(len(input_ids), node)
        canonical_indices = handle.get_matched_indices()[: len(input_ids)].clone()
        return DropAwareInsertResult(existing_prefix_len, handle, canonical_indices)

    def insert_prefix(
        self,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
        virtual_mask: torch.Tensor | None = None,
    ) -> InsertResult:
        if len(input_ids) != len(indices):
            raise ValueError("Radix keys and page indices must have equal lengths.")
        virtual_mask = self._normalize_virtual_mask(input_ids, virtual_mask)
        if self.drop_aware_eviction:
            key_to_token = torch.full(
                (len(input_ids),), -1, dtype=torch.int64, device="cpu"
            )
            real_positions = torch.nonzero(~virtual_mask, as_tuple=False).view(-1)
            key_to_token[real_positions] = torch.arange(
                len(real_positions), dtype=torch.int64, device="cpu"
            )
            keep_mask = torch.ones(len(real_positions), dtype=torch.bool, device="cpu")
            result = self.commit_drop_prefix(
                input_ids,
                indices,
                virtual_mask,
                key_to_token,
                keep_mask,
            )
            return InsertResult(result.existing_prefix_len, result.handle)
        value_virtual_mask = virtual_mask.to(device=indices.device, non_blocking=True)
        if bool(torch.any(indices[value_virtual_mask] != -1).item()):
            raise ValueError("Virtual Radix keys must use page value -1.")
        if bool(torch.any(indices[~value_virtual_mask] < 0).item()):
            raise ValueError("Real Radix keys must not contain negative page holes.")
        if self.page_size != 1:
            real_len = int(torch.count_nonzero(~virtual_mask).item())
            if real_len % self.page_size != 0:
                raise ValueError("Real Radix keys require page alignment for page_size > 1.")

        insert_len = (
            len(input_ids)
            if bool(torch.any(virtual_mask).item())
            else align_down(len(input_ids), self.page_size)
        )
        input_ids = input_ids[:insert_len]
        indices = indices[:insert_len]
        virtual_mask = virtual_mask[:insert_len]
        node, prefix_len = self._tree_walk(input_ids, virtual_mask)
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(
                input_ids[prefix_len:],
                indices[prefix_len:].clone(),
                virtual_mask[prefix_len:].clone(),
            )
            new_node.set_parent(node)
            self.evictable_size += new_node.page_length
            if self.delta_marker_registry is not None:
                self.delta_marker_registry.add_tree_refs(
                    self._marker_ids(new_node._key, new_node.virtual_mask)
                )
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def prune_suffix(
        self,
        input_ids: torch.Tensor,
        valid_prefix_len: int,
        virtual_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Remove an unprotected legacy sparse branch after its first invalid slot."""

        if self.drop_aware_eviction:
            raise RuntimeError("Drop-aware Radix holes preserve their structural suffix.")
        virtual_mask = self._normalize_virtual_mask(input_ids, virtual_mask)
        if self.page_size != 1:
            raise RuntimeError("Legacy sparse Radix pruning is only supported for page_size=1.")
        if valid_prefix_len < 0 or valid_prefix_len >= len(input_ids):
            raise ValueError("The prune boundary must point inside the matched path.")

        boundary = self.root_node
        cursor = 0
        while cursor < valid_prefix_len:
            child = boundary.children.get(
                _edge_key(self.key_fn, input_ids[cursor:], virtual_mask[cursor:])
            )
            if child is None:
                raise RuntimeError("Cannot prune a Radix path that is not present.")
            span = min(child.length, valid_prefix_len - cursor)
            if not torch.equal(
                child._key[:span], input_ids[cursor : cursor + span]
            ) or not torch.equal(child.virtual_mask[:span], virtual_mask[cursor : cursor + span]):
                raise RuntimeError("Radix path changed while locating the prune boundary.")
            if span < child.length:
                if child.ref_count > 0:
                    return None
                boundary = child.split_at(span)
            else:
                boundary = child
            cursor += span

        stale_key = _edge_key(
            self.key_fn,
            input_ids[valid_prefix_len:],
            virtual_mask[valid_prefix_len:],
        )
        stale_root = boundary.children.get(stale_key)
        if stale_root is None:
            raise RuntimeError("Cannot find the stale Radix suffix to prune.")

        stack = [stale_root]
        subtree: List[RadixTreeNode] = []
        while stack:
            node = stack.pop()
            if node.ref_count > 0:
                return None
            subtree.append(node)
            stack.extend(node.children.values())

        page_count = sum(node.page_length for node in subtree)
        pages = [
            node.value[(~node.virtual_mask).to(device=node.value.device, non_blocking=True)]
            for node in subtree
            if node.page_length > 0
        ]
        del boundary.children[stale_key]
        if self.delta_marker_registry is not None:
            for stale_node in subtree:
                self.delta_marker_registry.remove_tree_refs(
                    self._marker_ids(stale_node._key, stale_node.virtual_mask)
                )
        self.evictable_size -= page_count
        if self.evictable_size < 0:
            raise RuntimeError("Radix evictable-size accounting underflow during prune.")

        if not pages:
            return self.empty_tensor
        return torch.cat(pages)

    def evict(self, size: int) -> torch.Tensor:
        if self.drop_aware_eviction:
            return self._evict_drop_aware(size)
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.page_length
            if node.page_length > 0:
                real_mask = (~node.virtual_mask).to(device=node.value.device, non_blocking=True)
                evicted_indices.append(node.value[real_mask])
            self.evictable_size -= node.page_length
            parent = node.parent
            del parent.children[_edge_key(self.key_fn, node._key, node.virtual_mask)]
            if self.delta_marker_registry is not None:
                self.delta_marker_registry.remove_tree_refs(
                    self._marker_ids(node._key, node.virtual_mask)
                )
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        if len(evicted_indices) == 0:
            return self.empty_tensor
        return torch.cat(evicted_indices)

    def _set_node_hole(self, node: RadixTreeNode) -> torch.Tensor:
        if not node.resident or node.kv_pin_count != 0:
            raise RuntimeError("Only resident, unpinned Radix nodes can become holes.")
        self._remove_size(node)
        real_mask = (~node.virtual_mask).to(device=node.value.device, non_blocking=True)
        released = node.value[real_mask].clone()
        self._unregister_node_slots(node)
        values = node.value.clone()
        values[real_mask] = -1
        node.set_key_value(node._key, values, node.virtual_mask)
        self._add_size(node)
        return released

    def _delete_drop_aware_leaf(
        self, node: RadixTreeNode
    ) -> tuple[torch.Tensor, RadixTreeNode]:
        if node.is_root() or not node.is_leaf() or node.ref_count != 0:
            raise RuntimeError("Drop-aware ordinary eviction requires an unreferenced leaf.")
        if node.kv_pin_count != 0:
            raise RuntimeError("An unreferenced Radix leaf still owns pinned KV slots.")
        self._adjust_leaf_contribution(node, -1)
        self._remove_size(node)
        released = self.empty_tensor
        if node.resident:
            real_mask = (~node.virtual_mask).to(device=node.value.device, non_blocking=True)
            released = node.value[real_mask].clone()
            self._unregister_node_slots(node)
        parent = node.parent
        del parent.children[_edge_key(self.key_fn, node._key, node.virtual_mask)]
        if self.delta_marker_registry is not None:
            self.delta_marker_registry.remove_tree_refs(
                self._marker_ids(node._key, node.virtual_mask)
            )
        if parent.is_leaf() and not parent.is_root():
            self._adjust_leaf_contribution(parent, 1)
        return released, parent

    def _evict_drop_aware(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        if size > self.evictable_size:
            raise RuntimeError(
                f"Cannot evict {size}, only {self.evictable_size} is evictable"
            )

        nodes = [self.root_node]
        drop_candidates: List[RadixTreeNode] = []
        ordinary_leaves: List[RadixTreeNode] = []
        while nodes:
            node = nodes.pop()
            if not node.is_root() and self._is_drop_reclaimable(node):
                drop_candidates.append(node)
            if node.is_leaf():
                if not node.is_root() and node.ref_count == 0:
                    ordinary_leaves.append(node)
            else:
                nodes.extend(node.children.values())

        heapq.heapify(drop_candidates)
        heapq.heapify(ordinary_leaves)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        # Preserve the Radix keys first: turn Drop-safe KV blocks into explicit holes.
        while drop_candidates and evicted_size < size:
            node = heapq.heappop(drop_candidates)
            if not self._is_drop_reclaimable(node):
                continue
            released = self._set_node_hole(node)
            if len(released) > 0:
                evicted_indices.append(released)
                evicted_size += len(released)

        # Fall back to the original unreferenced-leaf LRU policy.
        while evicted_size < size:
            if not ordinary_leaves:
                raise RuntimeError(
                    f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
                )
            node = heapq.heappop(ordinary_leaves)
            if node.is_root() or not node.is_leaf() or node.ref_count != 0:
                continue
            released, parent = self._delete_drop_aware_leaf(node)
            if len(released) > 0:
                evicted_indices.append(released)
                evicted_size += len(released)
            if parent.is_leaf() and not parent.is_root() and parent.ref_count == 0:
                heapq.heappush(ordinary_leaves, parent)

        if not evicted_indices:
            return self.empty_tensor
        return torch.cat(evicted_indices)

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    @property
    def resident_slots(self) -> frozenset[int]:
        if not self.drop_aware_eviction:
            raise RuntimeError("Resident-slot indexing is only enabled for Drop-aware eviction.")
        return frozenset(self._slot_owner)

    def check_integrity(self) -> None:
        expected_evictable = 0
        expected_protected = 0
        actual_marker_refs: Counter[int] = Counter()
        stack: List[RadixTreeNode] = [self.root_node]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                value_virtual_mask = child.virtual_mask.to(
                    device=child.value.device, non_blocking=True
                )
                if bool(torch.any(child.value[value_virtual_mask] != -1).item()):
                    raise RuntimeError(
                        "RadixPrefixCache integrity check failed: a virtual key does not use -1."
                    )
                real_values = child.value[~value_virtual_mask]
                has_real_hole = bool(torch.any(real_values < 0).item())
                if has_real_hole and not self.drop_aware_eviction:
                    raise RuntimeError(
                        "RadixPrefixCache integrity check failed: a real key has a negative page."
                    )
                actual_marker_refs.update(self._marker_ids(child._key, child.virtual_mask))
                resident_size = (
                    child.resident_page_length
                    if self.drop_aware_eviction
                    else child.page_length
                )
                if self._is_evictable(child):
                    expected_evictable += resident_size
                else:
                    expected_protected += resident_size
                stack.append(child)
        if expected_evictable != self.evictable_size or expected_protected != self.protected_size:
            raise RuntimeError(
                "RadixPrefixCache integrity check failed:"
                f" evictable({self.evictable_size}) != expected({expected_evictable}) or"
                f" protected({self.protected_size}) != expected({expected_protected})"
            )
        if self.delta_marker_registry is not None:
            self.delta_marker_registry.check_tree_refs(actual_marker_refs)
        if self.drop_aware_eviction:
            self._check_drop_aware_integrity()

    def _check_drop_aware_integrity(self) -> None:
        expected_owners: Dict[int, RadixTreeNode] = {}
        expected_need_counts: Counter[int] = Counter()
        leaves: List[RadixTreeNode] = []
        all_nodes: List[RadixTreeNode] = []
        nodes = [self.root_node]
        while nodes:
            node = nodes.pop()
            if not node.is_root():
                all_nodes.append(node)
                has_virtual = bool(torch.any(node.virtual_mask).item())
                has_real = node.page_length > 0
                if has_virtual and has_real:
                    raise RuntimeError("A Drop-aware node mixes marker and real keys.")
                real_mask = (~node.virtual_mask).to(
                    device=node.value.device, non_blocking=True
                )
                real_values = node.value[real_mask]
                all_resident = len(real_values) > 0 and bool(
                    torch.all(real_values >= 0).item()
                )
                all_holes = len(real_values) == 0 or bool(
                    torch.all(real_values == -1).item()
                )
                if not (all_resident or all_holes) or node.resident != all_resident:
                    raise RuntimeError("A Drop-aware node has mixed or stale residency state.")
                if node.kv_pin_count < 0 or node.kv_need_leaf_count < 0:
                    raise RuntimeError("A Drop-aware node has a negative counter.")
                if node.resident:
                    for slot in self._real_slots(node):
                        if slot in expected_owners:
                            raise RuntimeError(f"KV slot {slot} is owned by multiple nodes.")
                        expected_owners[slot] = node
            if node.is_leaf():
                if not node.is_root():
                    leaves.append(node)
            else:
                nodes.extend(node.children.values())

        for leaf in leaves:
            keep_mask = self._leaf_keep_mask(leaf)
            token_cursor = 0
            for node in self._path_nodes(leaf):
                if node.page_length == 0:
                    continue
                segment = keep_mask[token_cursor : token_cursor + node.page_length]
                all_kept = bool(torch.all(segment).item())
                all_dropped = bool(torch.all(~segment).item())
                if not (all_kept or all_dropped):
                    raise RuntimeError("A leaf Drop range cuts through a Radix node.")
                if all_kept:
                    expected_need_counts[node.uuid] += 1
                token_cursor += node.page_length

        nodes = [self.root_node]
        while nodes:
            node = nodes.pop()
            if not node.is_root() and node.kv_need_leaf_count != expected_need_counts[node.uuid]:
                raise RuntimeError(
                    "Radix KV-need leaf count mismatch: "
                    f"node={node.uuid}, actual={node.kv_need_leaf_count}, "
                    f"expected={expected_need_counts[node.uuid]}"
                )
            nodes.extend(node.children.values())

        if expected_owners != self._slot_owner:
            raise RuntimeError("Drop-aware Radix slot-owner index is inconsistent.")
        expected_pin_counts: Counter[int] = Counter()
        for slot, count in self._slot_pin_count.items():
            if count < 0 or (count > 0 and slot not in self._slot_owner):
                raise RuntimeError(f"Invalid KV slot pin state: slot={slot}, count={count}")
            if count:
                expected_pin_counts[slot] = count
        if Counter({slot: count for slot, count in self._slot_pin_count.items() if count}) != (
            expected_pin_counts
        ):
            raise RuntimeError("Drop-aware KV slot pin index is inconsistent.")
        expected_node_pins: Counter[int] = Counter()
        for slot, count in expected_pin_counts.items():
            expected_node_pins[self._slot_owner[slot].uuid] += count
        for node in all_nodes:
            if node.kv_pin_count != expected_node_pins[node.uuid]:
                raise RuntimeError(
                    f"Radix KV pin count mismatch for node {node.uuid}: "
                    f"actual={node.kv_pin_count}, expected={expected_node_pins[node.uuid]}"
                )

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    @staticmethod
    def _normalize_virtual_mask(
        input_ids: torch.Tensor, virtual_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if virtual_mask is None:
            return torch.zeros(len(input_ids), dtype=torch.bool, device="cpu")
        if (
            virtual_mask.device.type != "cpu"
            or virtual_mask.dtype != torch.bool
            or virtual_mask.ndim != 1
        ):
            raise ValueError("Radix virtual_mask must be a CPU 1D bool tensor.")
        if len(virtual_mask) != len(input_ids):
            raise ValueError("Radix keys and virtual_mask must have equal lengths.")
        return virtual_mask

    def _tree_walk(
        self, input_ids: torch.Tensor, virtual_mask: torch.Tensor
    ) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            child_node = node.children.get(
                _edge_key(
                    self.key_fn,
                    input_ids[prefix_len:],
                    virtual_mask[prefix_len:],
                )
            )
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:], virtual_mask[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = (
                    self._split_node(node, match_len)
                    if self.drop_aware_eviction
                    else node.split_at(match_len)
                )
                node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
