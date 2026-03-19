import torch


class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

        # Per-table-idx metadata for reuse
        self._ref_count = [0] * max_running_reqs       # 引用计数
        self._transferred = [False] * max_running_reqs  # 是否已转移

        # reuse_key (hash(uid)) -> table_idx
        self._reuse_key_map: dict[int, int] = {}
        # table_idx -> cached_len (有效 page 数)
        self._reuse_cached_len: dict[int, int] = {}

        # Track evictable size
        self._evictable_size = 0

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)

    def get_reuse_info(self, reuse_key: int):
        """Look up the old table_idx for reuse.

        Returns:
            tuple[int, int] | None: (old_table_idx, cached_len) or None
        """
        if reuse_key not in self._reuse_key_map:
            return None
        old_table_idx = self._reuse_key_map[reuse_key]
        cached_len = self._reuse_cached_len[old_table_idx]
        return old_table_idx, cached_len

    def mark_as_transferred(self, old_table_idx: int, new_table_idx: int,
                             new_cached_len: int, reuse_key: int) -> None:
        """Mark old table as transferred after restore extracts its data.

        Called during restore phase in prefill.
        - Increments ref_count on old table (active request using its indices)
        - Marks old table as transferred
        - Removes old table from evictable tracking
        - Removes old mapping from reuse_key_map
        """
        self._ref_count[old_table_idx] += 1
        self._transferred[old_table_idx] = True

        # Remove old table from evictable size
        old_cached_len = self._reuse_cached_len.pop(old_table_idx, 0)
        self._evictable_size -= old_cached_len

        # Remove old mapping (request still in flight, will be re-added in store_for_reuse)
        if reuse_key in self._reuse_key_map:
            del self._reuse_key_map[reuse_key]

    def store_for_reuse(self, table_idx: int, cached_len: int, reuse_key: int,
                         old_table_idx: int | None = None) -> None:
        """Mark a table_idx as reusable after request completes.

        Args:
            table_idx: Current table to preserve for future reuse
            cached_len: Number of valid indices
            reuse_key: Session identifier
            old_table_idx: The old table that was transferred (if any)
        """
        # Decrement ref_count on old table and reclaim if possible
        if old_table_idx is not None:
            self._ref_count[old_table_idx] -= 1
            if self._ref_count[old_table_idx] == 0 and self._transferred[old_table_idx]:
                # Reclaim old table slot (indices already transferred, don't evict)
                self._transferred[old_table_idx] = False
                self._free_slots.append(old_table_idx)

        # Register new table for reuse
        self._reuse_key_map[reuse_key] = table_idx
        self._reuse_cached_len[table_idx] = cached_len
        self._evictable_size += cached_len

    def evict_one(self) -> torch.Tensor | None:
        """Evict one reusable table_idx (ref_count==0, not transferred).
        Returns its page indices for freeing."""
        for reuse_key, table_idx in list(self._reuse_key_map.items()):
            if self._ref_count[table_idx] == 0 and not self._transferred[table_idx]:
                cached_len = self._reuse_cached_len.pop(table_idx, 0)
                indices = self.page_table[table_idx, :cached_len].clone()
                self._evictable_size -= cached_len
                del self._reuse_key_map[reuse_key]
                self._free_slots.append(table_idx)
                return indices
        return None

    def get_occupied_pages(self) -> int:
        """Return total pages in evictable storage (non-transferred, tracked)."""
        return self._evictable_size
