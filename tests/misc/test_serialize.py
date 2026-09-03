from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from minisgl.core import SamplingParams
from minisgl.message import (
    BaseBackendMsg,
    BaseFrontendMsg,
    BatchBackendMsg,
    RepositionOpenMsg,
    UserMsg,
    UserReply,
    WarmupAckMsg,
)
from minisgl.message.utils import deserialize_type, serialize_type
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


@dataclass
class A:
    x: int
    y: str
    z: List[A]
    w: torch.Tensor


@call_if_main()
def test_serialize_deserialize():

    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    x = A(10, "hello", [A(20, "world", [], t)], t)
    data = serialize_type(x)
    logger.info(data)
    y = deserialize_type({"A": A}, data)
    logger.info(y)

    u = BatchBackendMsg(
        [
            UserMsg(
                uid=0,
                input_ids=t,
                true_positions=torch.arange(len(t), dtype=torch.int32),
                raw_positions=torch.arange(len(t), dtype=torch.int32),
                radix_input_ids=t.to(torch.int64),
                sampling_params=SamplingParams(),
                drop_effective_event_count=2,
            )
        ]
    )
    result = u.decoder(u.encoder())
    logger.info(u)
    logger.info(result)
    assert result.data[0].drop_effective_event_count == 2

    records = torch.tensor(
        [[0, 10, -1, 0], [1, -1, -2, -1], [0, 11, 0, 1]],
        dtype=torch.int32,
    )
    structured = UserMsg(
        uid=1,
        input_ids=t,
        true_positions=torch.arange(len(t), dtype=torch.int32),
        raw_positions=torch.arange(len(t), dtype=torch.int32),
        radix_input_ids=records,
        radix_match_ids=records,
        radix_key_virtual_mask=torch.tensor([False, True, False]),
        sampling_params=SamplingParams(),
    )
    restored_structured = BaseBackendMsg.decoder(structured.encoder())
    assert torch.equal(restored_structured.radix_match_ids, records)
    assert restored_structured.radix_key_virtual_mask.tolist() == [False, True, False]

    open_msg = RepositionOpenMsg(uid=2)
    restored_open = BaseBackendMsg.decoder(open_msg.encoder())
    assert vars(restored_open) == {"uid": 2}

    warmup = WarmupAckMsg(
        uid=3,
        hit_ratio=0.5,
        cached_tokens=4,
        drop_skipped_tokens=7,
        finished=True,
    )
    restored = WarmupAckMsg.decoder(warmup.encoder(warmup))
    assert restored.drop_skipped_tokens == 7

    reply = UserReply(
        uid=4,
        incremental_output="answer",
        finished=True,
        incremental_token_ids=[10, 11],
    )
    restored_reply = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(reply))
    assert restored_reply.incremental_token_ids == [10, 11]

    legacy_reply = BaseFrontendMsg.encoder(reply)
    legacy_reply.pop("incremental_token_ids")
    restored_legacy_reply = BaseFrontendMsg.decoder(legacy_reply)
    assert restored_legacy_reply.incremental_token_ids == []
