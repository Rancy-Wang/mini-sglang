from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
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
        self.children_exact: dict[int, list[RadixTreeNode]] = {}
        self.children_retry: dict[int, list[RadixTreeNode]] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()
        self.max_reachable_depth = 0

        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._virtual_mask: torch.Tensor
        self._length: int
        self._page_length: int

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
        self._refresh_reachable_depth()

    def set_parent(self, parent: RadixTreeNode) -> None:
        previous_parent = self._parent
        if previous_parent is not None:
            previous_parent._remove_child(self)
            previous_parent._refresh_reachable_depth()
        self._parent = parent
        parent._add_child(self)
        self._refresh_reachable_depth()
        parent._refresh_reachable_depth()

    def _add_child(self, child: RadixTreeNode) -> None:
        if child._key.ndim != 2:
            self.children[_edge_key(self.key_fn, child._key, child._virtual_mask)] = child
            return
        from minisgl.kernel.radix import radix_record_edge_hash, radix_record_retry_token

        self.children[("record", child.uuid)] = child
        edge_hash = radix_record_edge_hash(child._key)
        self.children_exact.setdefault(edge_hash, []).append(child)
        retry_token = radix_record_retry_token(child._key)
        if retry_token >= 0:
            self.children_retry.setdefault(retry_token, []).append(child)

    def _remove_child(self, child: RadixTreeNode) -> None:
        if child._key.ndim != 2:
            del self.children[_edge_key(self.key_fn, child._key, child._virtual_mask)]
            return
        from minisgl.kernel.radix import radix_record_edge_hash, radix_record_retry_token

        del self.children[("record", child.uuid)]
        edge_hash = radix_record_edge_hash(child._key)
        exact = self.children_exact[edge_hash]
        exact.remove(child)
        if not exact:
            del self.children_exact[edge_hash]
        retry_token = radix_record_retry_token(child._key)
        if retry_token >= 0:
            retry = self.children_retry[retry_token]
            retry.remove(child)
            if not retry:
                del self.children_retry[retry_token]

    def find_exact_child(
        self, key: torch.Tensor, virtual_mask: torch.Tensor
    ) -> RadixTreeNode | None:
        if key.ndim != 2:
            return self.children.get(_edge_key(self.key_fn, key, virtual_mask))
        from minisgl.kernel.radix import radix_record_edge_equal, radix_record_edge_hash

        for child in self.children_exact.get(radix_record_edge_hash(key), ()):
            if radix_record_edge_equal(child._key, key):
                return child
        return None

    def retry_children(self, target: torch.Tensor) -> tuple[RadixTreeNode, ...]:
        from minisgl.kernel.radix import radix_record_retry_token

        retry_token = radix_record_retry_token(target)
        if retry_token >= 0:
            return tuple(self.children_retry.get(retry_token, ()))
        child = self.find_exact_child(
            target,
            torch.empty(0, dtype=torch.bool, device="cpu"),
        )
        return () if child is None else (child,)

    def _refresh_reachable_depth(self) -> None:
        own_length = 0 if self.is_root() else getattr(self, "_length", 0)
        child_depth = max(
            (child.max_reachable_depth for child in self.children.values()),
            default=0,
        )
        updated = own_length + child_depth
        if updated == self.max_reachable_depth:
            return
        self.max_reachable_depth = updated
        if self._parent is not None:
            self._parent._refresh_reachable_depth()

    @property
    def length(self) -> int:
        return self._length

    @property
    def page_length(self) -> int:
        return self._page_length

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
        from minisgl.kernel.radix import fast_compare_radix_key, fast_compare_radix_records

        if self._key.ndim == 2:
            return fast_compare_radix_records(self._key, input_ids)
        return fast_compare_radix_key(self._key, input_ids, self._virtual_mask, virtual_mask)

    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent
        parent._remove_child(self)
        self._parent = None

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos], self._virtual_mask[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:], self._virtual_mask[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

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

    def get_matched_keys(self) -> torch.Tensor:
        node = self.node
        keys: List[torch.Tensor] = []
        while not node.is_root():
            keys.append(node._key)
            node = node.parent
        keys.reverse()
        return torch.cat(keys)

    @property
    def physical_cached_len(self) -> int:
        node = self.node
        length = 0
        while not node.is_root():
            length += node.page_length
            node = node.parent
        return length


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
        self._ordinary_slot_nodes: Dict[int, set[RadixTreeNode]] = {}

    def _real_slots(self, node: RadixTreeNode) -> List[int]:
        real_mask = (~node.virtual_mask).to(device=node.value.device, non_blocking=True)
        return [int(slot) for slot in node.value[real_mask].tolist()]

    def _split_node(self, node: RadixTreeNode, pos: int) -> RadixTreeNode:
        prefix_slots = self._real_slots_from_slice(node, slice(0, pos))
        new_node = node.split_at(pos)
        for slot in prefix_slots:
            owners = self._ordinary_slot_nodes.get(slot)
            if owners is None or node not in owners:
                raise RuntimeError(f"Radix shared-slot owner mismatch while splitting {slot}.")
            owners.remove(node)
            owners.add(new_node)
        return new_node

    @staticmethod
    def _real_slots_from_slice(node: RadixTreeNode, span: slice) -> List[int]:
        values = node.value[span]
        virtual = node.virtual_mask[span]
        real_mask = (~virtual).to(device=values.device, non_blocking=True)
        return [int(slot) for slot in values[real_mask].tolist()]

    def _register_ordinary_node(self, node: RadixTreeNode) -> None:
        for slot in self._real_slots(node):
            owners = self._ordinary_slot_nodes.setdefault(slot, set())
            if node in owners:
                raise RuntimeError(f"Radix node registered KV slot {slot} twice.")
            was_protected = any(owner.ref_count > 0 for owner in owners)
            if not owners:
                if node.ref_count > 0:
                    self.protected_size += 1
                else:
                    self.evictable_size += 1
            owners.add(node)
            if not was_protected and node.ref_count > 0 and len(owners) > 1:
                self.evictable_size -= 1
                self.protected_size += 1

    def _unregister_ordinary_node(self, node: RadixTreeNode) -> torch.Tensor:
        released: list[int] = []
        for slot in self._real_slots(node):
            owners = self._ordinary_slot_nodes.get(slot)
            if owners is None or node not in owners:
                raise RuntimeError(f"Radix shared-slot owner mismatch while releasing {slot}.")
            was_protected = any(owner.ref_count > 0 for owner in owners)
            owners.remove(node)
            if not owners:
                del self._ordinary_slot_nodes[slot]
                if was_protected:
                    self.protected_size -= 1
                else:
                    self.evictable_size -= 1
                released.append(slot)
                continue
            is_protected = any(owner.ref_count > 0 for owner in owners)
            if was_protected and not is_protected:
                self.protected_size -= 1
                self.evictable_size += 1
        if not released:
            return self.empty_tensor
        return torch.tensor(released, dtype=torch.int32, device=self.device)

    def _ordinary_node_became_protected(self, node: RadixTreeNode) -> None:
        for slot in set(self._real_slots(node)):
            owners = self._ordinary_slot_nodes[slot]
            if not any(owner is not node and owner.ref_count > 0 for owner in owners):
                self.evictable_size -= 1
                self.protected_size += 1

    def _ordinary_node_became_evictable(self, node: RadixTreeNode) -> None:
        for slot in set(self._real_slots(node)):
            owners = self._ordinary_slot_nodes[slot]
            if not any(owner is not node and owner.ref_count > 0 for owner in owners):
                self.protected_size -= 1
                self.evictable_size += 1

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                if node.ref_count == 1:
                    self._ordinary_node_became_evictable(node)
                node.ref_count -= 1
                assert node.ref_count >= 0
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self._ordinary_node_became_protected(node)
                node.ref_count += 1
                node = node.parent

    def match_prefix(
        self, input_ids: torch.Tensor, virtual_mask: torch.Tensor | None = None
    ) -> MatchResult:
        virtual_mask = self._normalize_virtual_mask(input_ids, virtual_mask)
        node, prefix_len = self._tree_walk(input_ids, virtual_mask)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def match_retry_prefix(
        self,
        target: torch.Tensor,
        virtual_mask: torch.Tensor,
        exact_handle: RadixCacheHandle,
    ) -> RadixCacheHandle:
        """Greedily extend one structured source branch after exact matching."""

        from minisgl.kernel.radix import fast_compare_retry_radix_records

        if target.ndim != 2 or target.shape[1] != 4:
            return exact_handle
        node = exact_handle.node
        cursor = exact_handle.cached_len
        tic = time.monotonic_ns()
        while cursor < len(target):
            candidates = node.retry_children(target[cursor:])
            if not candidates:
                break
            child = max(
                candidates,
                key=lambda candidate: (candidate.max_reachable_depth, -candidate.uuid),
            )
            match_len = fast_compare_retry_radix_records(child._key, target[cursor:])
            if match_len == 0:
                break
            cursor += match_len
            node = child
            if match_len != child.length:
                node = self._split_node(child, match_len)
                node.timestamp = tic
                break
            node.timestamp = tic
        return RadixCacheHandle(cursor, node)

    def insert_prefix(
        self,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
        virtual_mask: torch.Tensor | None = None,
    ) -> InsertResult:
        if len(input_ids) != len(indices):
            raise ValueError("Radix keys and page indices must have equal lengths.")
        virtual_mask = self._normalize_virtual_mask(input_ids, virtual_mask)
        value_virtual_mask = virtual_mask.to(device=indices.device, non_blocking=True)
        if bool(torch.any(indices[value_virtual_mask] != -1).item()):
            raise ValueError("Virtual Radix keys must use page value -1.")
        if bool(torch.any(indices[~value_virtual_mask] < 0).item()):
            raise ValueError("Real Radix keys must not contain negative page holes.")
        if self.page_size != 1 and bool(torch.any(virtual_mask).item()):
            raise ValueError("Virtual Radix keys require page_size=1.")

        insert_len = (
            len(input_ids)
            if bool(torch.any(virtual_mask).item())
            else align_down(len(input_ids), self.page_size)
        )
        input_ids = input_ids[:insert_len]
        indices = indices[:insert_len]
        virtual_mask = virtual_mask[:insert_len]
        node, prefix_len = self._tree_walk(input_ids, virtual_mask)
        existing_prefix_len = prefix_len
        if prefix_len != insert_len:
            segment_ends = (
                self._structured_segment_ends(input_ids, virtual_mask)
                if input_ids.ndim == 2
                else [insert_len]
            )
            for segment_end in segment_ends:
                if segment_end <= prefix_len:
                    continue
                new_node = RadixTreeNode(self.key_fn)
                new_node.set_key_value(
                    input_ids[prefix_len:segment_end],
                    indices[prefix_len:segment_end].clone(),
                    virtual_mask[prefix_len:segment_end].clone(),
                )
                new_node.set_parent(node)
                self._register_ordinary_node(new_node)
                node = new_node
                prefix_len = segment_end
        return InsertResult(existing_prefix_len, RadixCacheHandle(insert_len, node))

    @staticmethod
    def _structured_segment_ends(records: torch.Tensor, virtual_mask: torch.Tensor) -> List[int]:
        """Keep each complete Delta range block on one Radix child edge."""

        if len(records) == 0:
            return []
        from minisgl.kernel.radix_reposition import DELTA_KIND

        previous_virtual = virtual_mask[:-1]
        current_virtual = virtual_mask[1:]
        virtual_transition = previous_virtual != current_virtual
        adjacent_virtual = previous_virtual & current_virtual
        adjacent_delta = (records[:-1, 0] == DELTA_KIND) & (records[1:, 0] == DELTA_KIND)
        split = virtual_transition | (adjacent_virtual & ~adjacent_delta)
        boundaries = (torch.nonzero(split, as_tuple=False).view(-1) + 1).tolist()
        boundaries.append(len(records))
        return boundaries

    def evict(self, size: int) -> torch.Tensor:
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
            parent = node.parent
            parent._remove_child(node)
            parent._refresh_reachable_depth()
            released = self._unregister_ordinary_node(node)
            if len(released) > 0:
                evicted_indices.append(released)
                evicted_size += len(released)
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        if len(evicted_indices) == 0:
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

    def check_integrity(self) -> None:
        pass

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
            child_node = node.find_exact_child(
                input_ids[prefix_len:],
                virtual_mask[prefix_len:],
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
                node = self._split_node(node, match_len)
                node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
