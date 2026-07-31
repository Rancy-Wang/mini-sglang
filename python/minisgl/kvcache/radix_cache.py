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
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

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

    def get_match_len(
        self, input_ids: torch.Tensor, virtual_mask: torch.Tensor
    ) -> int:
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        key_match_len = fast_compare_key(self._key, input_ids)
        virtual_match = self._virtual_mask[:key_match_len] == virtual_mask[:key_match_len]
        mismatch = torch.nonzero(~virtual_match, as_tuple=False).view(-1)
        if len(mismatch) > 0:
            return int(mismatch[0].item())
        return key_match_len

    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(
            self._key[:pos], self._value[:pos], self._virtual_mask[:pos]
        )
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(
            self._key[pos:], self._value[pos:], self._virtual_mask[pos:]
        )
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

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
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
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(
                input_ids[prefix_len:],
                indices[prefix_len:].clone(),
                virtual_mask[prefix_len:].clone(),
            )
            new_node.set_parent(node)
            self.evictable_size += new_node.page_length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def prune_suffix(
        self,
        input_ids: torch.Tensor,
        valid_prefix_len: int,
        virtual_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Remove an unprotected legacy sparse branch after its first invalid slot."""

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
            ) or not torch.equal(
                child.virtual_mask[:span], virtual_mask[cursor : cursor + span]
            ):
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
            node.value[
                (~node.virtual_mask).to(device=node.value.device, non_blocking=True)
            ]
            for node in subtree
            if node.page_length > 0
        ]
        del boundary.children[stale_key]
        self.evictable_size -= page_count
        if self.evictable_size < 0:
            raise RuntimeError("Radix evictable-size accounting underflow during prune.")

        if not pages:
            return self.empty_tensor
        return torch.cat(pages)

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
            evicted_size += node.page_length
            if node.page_length > 0:
                real_mask = (~node.virtual_mask).to(
                    device=node.value.device, non_blocking=True
                )
                evicted_indices.append(node.value[real_mask])
            self.evictable_size -= node.page_length
            parent = node.parent
            del parent.children[_edge_key(self.key_fn, node._key, node.virtual_mask)]
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
        expected_evictable = 0
        expected_protected = 0
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
                if bool(torch.any(child.value[~value_virtual_mask] < 0).item()):
                    raise RuntimeError(
                        "RadixPrefixCache integrity check failed: a real key has a negative page."
                    )
                if child.ref_count == 0:
                    expected_evictable += child.page_length
                else:
                    expected_protected += child.page_length
                stack.append(child)
        if expected_evictable != self.evictable_size or expected_protected != self.protected_size:
            raise RuntimeError(
                "RadixPrefixCache integrity check failed:"
                f" evictable({self.evictable_size}) != expected({expected_evictable}) or"
                f" protected({self.protected_size}) != expected({expected_protected})"
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
            match_len = node.get_match_len(
                input_ids[prefix_len:], virtual_mask[prefix_len:]
            )
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
