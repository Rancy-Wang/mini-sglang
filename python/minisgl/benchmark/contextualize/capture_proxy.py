from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Iterable, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from .manifest import CaptureRecord, request_hash

_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _forward_headers(headers: Iterable[Tuple[str, str]]) -> dict[str, str]:
    return {key: value for key, value in headers if key.lower() not in _HOP_BY_HOP_HEADERS}


class CaptureStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()

    async def append(self, record: CaptureRecord) -> None:
        payload = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        async with self._lock:
            await asyncio.to_thread(self._append_line, payload)

    def _append_line(self, payload: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(payload + "\n")


def create_capture_app(
    *,
    upstream_base_url: str,
    output_path: str | Path,
    timeout_seconds: float = 3600.0,
) -> FastAPI:
    upstream = upstream_base_url.rstrip("/")
    store = CaptureStore(output_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.client = httpx.AsyncClient(timeout=timeout_seconds)
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(title="Contextualize request capture proxy", lifespan=lifespan)

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    async def proxy(path: str, request: Request):
        body = await request.body()
        normalized_path = "/" + path.lstrip("/")
        if normalized_path == "/v1/chat/completions" and request.method == "POST":
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Request body is not JSON: {exc}",
                ) from exc
            if not isinstance(payload, dict):
                raise HTTPException(status_code=400, detail="Request JSON must be an object.")
            await store.append(
                CaptureRecord(
                    capture_id=uuid.uuid4().hex,
                    captured_at_ns=time.time_ns(),
                    request=payload,
                    request_sha256=request_hash(payload),
                )
            )

        client: httpx.AsyncClient = request.app.state.client
        upstream_request = client.build_request(
            request.method,
            upstream + normalized_path,
            params=request.query_params,
            headers=_forward_headers(request.headers.items()),
            content=body,
        )
        response = await client.send(upstream_request, stream=True)
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=_forward_headers(response.headers.items()),
            background=BackgroundTask(response.aclose),
        )

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capture exact Contextualize OpenAI requests.")
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args(argv)
    app = create_capture_app(
        upstream_base_url=args.upstream_base_url,
        output_path=args.output,
        timeout_seconds=args.timeout,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
