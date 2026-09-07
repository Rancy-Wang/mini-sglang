from __future__ import annotations

from dataclasses import fields

import pytest
import torch

pytest.importorskip("tvm_ffi")

from minisgl.core import Req
from minisgl.kernel.radix_reposition import (
    DELTA_KIND,
    RadixRepositionInput,
    compile_radix_reposition_layout,
)
from minisgl.message import RepositionOpenMsg, UserMsg
from minisgl.message.tokenizer import RepositionOpenAckMsg
from minisgl.scheduler.radix_delta import decode_delta_record, validate_delta_records
from minisgl.scheduler.utils import PendingReq


def _compile() -> object:
    return compile_radix_reposition_layout(
        torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32),
        torch.tensor([6], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0, 2, 5, 6], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
    )


def test_compiler_embeds_each_half_open_range_without_allocated_ids() -> None:
    first = _compile()
    second = _compile()

    assert first.records[-2:].tolist() == [
        [DELTA_KIND, -1, -3, -1],
        [DELTA_KIND, -6, -7, -1],
    ]
    assert first.drop_event_to_key.tolist() == [6]
    assert torch.equal(first.records, second.records)
    assert [decode_delta_record(row) for row in first.records[-2:].tolist()] == [
        (0, 2),
        (5, 6),
    ]
    validate_delta_records(first.records, token_count=6, require_materialized=True)


@pytest.mark.parametrize(
    "record",
    (
        [DELTA_KIND, 0, -2, -1],
        [DELTA_KIND, -1, -1, -1],
        [DELTA_KIND, -3, -2, -1],
        [DELTA_KIND, -1, -2, 0],
    ),
)
def test_direct_delta_decoder_rejects_invalid_records(record: list[int]) -> None:
    with pytest.raises(ValueError):
        decode_delta_record(record)


def test_delta_record_cannot_precede_the_tokens_it_drops() -> None:
    records = torch.tensor(
        [[DELTA_KIND, -1, -3, -1], [0, 10, -1, 0], [0, 11, -1, 1]],
        dtype=torch.int32,
    )
    with pytest.raises(ValueError, match="materialization boundary"):
        validate_delta_records(records, token_count=2, require_materialized=True)


def test_protocol_and_request_state_have_no_allocated_delta_id_field() -> None:
    classes = (
        RadixRepositionInput,
        RepositionOpenMsg,
        RepositionOpenAckMsg,
        UserMsg,
        PendingReq,
        Req,
    )
    for cls in classes:
        assert "radix_marker_ids" not in {field.name for field in fields(cls)}
        assert "delta_marker_ids" not in {field.name for field in fields(cls)}

    assert {field.name for field in fields(RepositionOpenMsg)} == {"uid"}
    assert {field.name for field in fields(RepositionOpenAckMsg)} == {
        "uid",
        "step_token_budget",
    }
