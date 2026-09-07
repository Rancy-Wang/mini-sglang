from __future__ import annotations

import torch


def _causal_attention(values: torch.Tensor) -> torch.Tensor:
    """A deterministic two-token attention layer with identity V projection."""

    count = len(values)
    scores = torch.zeros((count, count), dtype=torch.float64)
    scores[torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)] = -torch.inf
    return scores.softmax(dim=-1) @ values


def test_deleting_a_kv_row_does_not_remove_its_indirect_effect_from_survivors():
    # At layer zero, token 1 attends to both token 0 and itself. Its next-layer
    # cached value therefore contains token 0 even after token 0's row is deleted.
    embeddings = torch.tensor([[4.0], [0.0]], dtype=torch.float64)
    full_layer_output = embeddings + _causal_attention(embeddings)
    stale_survivor_kv = full_layer_output[1:].clone()
    copied_survivor_kv = stale_survivor_kv.clone()

    # Rebuilding token 1 after token 0 is gone produces the true active-context KV.
    active_embeddings = embeddings[1:]
    rebuilt_survivor_kv = active_embeddings + _causal_attention(active_embeddings)

    assert torch.equal(copied_survivor_kv, stale_survivor_kv)
    assert stale_survivor_kv.item() == 2.0
    assert rebuilt_survivor_kv.item() == 0.0

    # A later query that can see only survivor token 1 still observes the stale
    # indirect contribution unless the survivor was recomputed.
    stale_query_output = _causal_attention(stale_survivor_kv)[-1]
    rebuilt_query_output = _causal_attention(rebuilt_survivor_kv)[-1]
    assert stale_query_output.item() == 2.0
    assert rebuilt_query_output.item() == 0.0
