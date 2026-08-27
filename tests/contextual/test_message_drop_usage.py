from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener


MESSAGES = [
    {"role": "system", "content": "You are a concise calculator."},
    {"role": "user", "content": "Compute 12 + 30."},
    {"role": "assistant", "content": "The result is 42."},
    {"role": "user", "content": "Now multiply it by 2."},
    {"role": "assistant", "content": "That gives 84."},
    {"role": "user", "content": "Remember the earlier inputs."},
    {"role": "assistant", "content": "I remember them."},
    {"role": "user", "content": "Give the final number only."},
]

PRIMER_DROP_RULE = {
    "type": "message_drop",
    "drop_messages": {"3": [0, 1]},
}

DROP_RULE = {
    "type": "message_drop",
    "drop_messages": {"3": [0, 1], "6": [2, 3]},
}

OPENER = build_opener(ProxyHandler({}))


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with OPENER.open(url, timeout=timeout) as response:
        return json.load(response)


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with OPENER.open(request, timeout=timeout) as response:
        return json.load(response)


def first_model(base_url: str, timeout: float) -> str:
    models = get_json(f"{base_url}/v1/models", timeout).get("data", [])
    if not models:
        raise RuntimeError("The server returned no models.")
    return str(models[0]["id"])


def send_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    drop_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if drop_rule is not None:
        payload["drop_rule"] = drop_rule
    return post_json(f"{base_url}/v1/chat/completions", payload, timeout)


def checked_usage(response: dict[str, Any]) -> dict[str, Any]:
    usage = response["usage"]
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    assert usage["total_tokens"] == prompt_tokens + completion_tokens

    details = usage.get("prompt_tokens_details")
    if details is not None:
        cached_tokens = details["cached_tokens"]
        skipped_tokens = details["drop_skipped_tokens"]
        assert 0 <= cached_tokens + skipped_tokens <= prompt_tokens
    return usage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--drop-primer", action="store_true")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    model = first_model(base_url, args.timeout)
    primer = send_chat(
        base_url,
        model,
        MESSAGES[:4],
        max_tokens=9,
        timeout=args.timeout,
        drop_rule=PRIMER_DROP_RULE if args.drop_primer else None,
    )
    target = send_chat(
        base_url,
        model,
        MESSAGES,
        max_tokens=8,
        timeout=args.timeout,
        drop_rule=DROP_RULE,
    )

    print(
        json.dumps(
            {
                "model": model,
                "primer_drop": args.drop_primer,
                "request1_usage": checked_usage(primer),
                "request2_usage": checked_usage(target),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
