import torch


class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)
        # Raw history is not bounded by the post-Reposition model position.
        # Keep graph-visible active/decode tables fixed; only overflowing
        # occurrence requests own separate prefill storage.
        self._occurrence_storage: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def prepare_occurrence(self, slot: int, raw_length: int) -> None:
        if raw_length < 1:
            raise ValueError("Occurrence raw length must be positive.")
        if raw_length + 1 <= self.page_table.shape[1]:
            return
        if slot in self._occurrence_storage:
            raise RuntimeError("Occurrence storage is already allocated.")
        pages = torch.full(
            (raw_length,), -1, dtype=torch.int32, device=self.page_table.device
        )
        tokens = torch.zeros(raw_length + 1, dtype=torch.int32, device=self.page_table.device)
        self._occurrence_storage[slot] = (pages, tokens)

    def occurrence_pages(self, slot: int) -> torch.Tensor:
        storage = self._occurrence_storage.get(slot)
        return self.page_table[slot] if storage is None else storage[0]

    def occurrence_tokens(self, slot: int) -> torch.Tensor:
        storage = self._occurrence_storage.get(slot)
        return self.token_pool[slot] if storage is None else storage[1]

    def has_occurrence_storage(self, slot: int) -> bool:
        return slot in self._occurrence_storage

    def release_occurrence(self, slot: int) -> None:
        self._occurrence_storage.pop(slot, None)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self.release_occurrence(slot)
        self._free_slots.append(slot)
