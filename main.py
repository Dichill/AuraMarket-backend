"""
AuraMarket backend (FastAPI on Google Cloud Run).

Endpoints
---------
GET  /           Service status and the Gemini models in use.
GET  /health     Liveness probe.
POST /scan       Frames sampled from the seller's video (or a short video file) plus an
                 optional product hint. Responds with an NDJSON stream of progress
                 events followed by one ``result`` (or ``error``) event.
GET  /model.glb  Rebuilds a generated 3D model from its recipe token and target size
                 in centimeters: ``/model.glb?v=1&w=72&h=80&d=70&r=<token>``.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from google import genai

import model_builder
from media import MediaError, extract_frames, normalize_frame
from pipeline import Event, PipelineConfig, ScanPipeline

load_dotenv(Path(__file__).with_name(".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("auramarket")

# Upload limits. Cloud Run rejects HTTP/1 request bodies above 32 MiB, so raw videos must
# stay below that; the web app normally sends ~20 small JPEG frames instead.
MAX_FRAMES = 24
MAX_FRAME_BYTES = 6 * 1024 * 1024
MAX_VIDEO_BYTES = 30 * 1024 * 1024
VIDEO_SAMPLE_FRAMES = 20
MAX_HINT_CHARS = 200
PING_INTERVAL_S = 10.0
READ_CHUNK_BYTES = 1024 * 1024

api_key = os.environ.get("GEMINI_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing!")

pipeline = ScanPipeline(genai.Client(api_key=api_key), PipelineConfig.from_env())

app = FastAPI(title="AuraMarket Backend", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, object]:
    """Service status and configured models."""
    return {"status": "AuraMarket Backend is running!", "version": app.version, "models": pipeline.config.as_dict()}


@app.get("/health")
def health() -> dict[str, bool]:
    """Liveness probe."""
    return {"ok": True}


async def _read_limited(upload: UploadFile, limit: int, label: str) -> bytes:
    """
    Read an upload into memory, rejecting it once it exceeds ``limit`` bytes.

    Raises:
        HTTPException: 413 when the file is too large.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"{label} is larger than {limit // (1024 * 1024)} MB.")
        chunks.append(chunk)
    return b"".join(chunks)


def _encode_event(event: Event) -> bytes:
    """Serialize one NDJSON line."""
    return f"{json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n".encode("utf-8")


async def _scan_stream(raw_frames: list[bytes], raw_video: bytes | None, video_name: str, hint: str) -> AsyncIterator[bytes]:
    """
    Run the pipeline in a background task and stream its events as NDJSON.

    A ``ping`` line is sent whenever the pipeline is quiet for ``PING_INTERVAL_S`` so
    proxies keep the connection open during the slower Gemini steps.
    """
    queue: asyncio.Queue[Event | None] = asyncio.Queue()

    async def emit(event: Event) -> None:
        await queue.put(event)

    async def run() -> None:
        try:
            await emit({"type": "progress", "stage": "frames", "status": "running", "message": "Reading your video..."})
            if raw_frames:
                frames = await asyncio.to_thread(lambda: [normalize_frame(frame) for frame in raw_frames])
            elif raw_video is not None:
                frames = await asyncio.to_thread(extract_frames, raw_video, video_name, VIDEO_SAMPLE_FRAMES)
            else:
                raise MediaError("No video frames were uploaded.")
            await emit({"type": "progress", "stage": "frames", "status": "done", "message": f"Using {len(frames)} frames from your video"})
            await pipeline.run_scan(frames, hint, emit)
        except MediaError as error:
            await emit({"type": "progress", "stage": "frames", "status": "failed", "message": str(error)})
            await emit({"type": "error", "stage": "frames", "message": str(error)})
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - report unexpected failures to the client instead of hanging
            logger.exception("Scan crashed")
            await emit({"type": "error", "stage": "internal", "message": f"Unexpected server error: {type(error).__name__}: {error}"})
        finally:
            await queue.put(None)

    task = asyncio.create_task(run())
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=PING_INTERVAL_S)
            except asyncio.TimeoutError:
                yield _encode_event({"type": "ping"})
                continue
            if event is None:
                break
            yield _encode_event(event)
    finally:
        # The client disconnected or the stream finished: never leave Gemini calls running.
        if not task.done():
            task.cancel()


@app.post("/scan")
async def scan(
    frames: list[UploadFile] | None = File(default=None, description="JPEG/PNG frames sampled in order from the video."),
    video: UploadFile | None = File(default=None, description="Alternative to frames: a video file up to 30 MB."),
    product_hint: str = Form(default="", description="Optional seller description, e.g. 'Sony Bravia 55 inch TV'."),
) -> StreamingResponse:
    """Scan a video of an item: identify, research, photograph and model it in 3D."""
    frame_uploads = frames or []
    if not frame_uploads and video is None:
        raise HTTPException(status_code=400, detail="Upload a video or frames from a video.")
    if len(frame_uploads) > MAX_FRAMES:
        raise HTTPException(status_code=400, detail=f"Send at most {MAX_FRAMES} frames.")

    raw_frames = [await _read_limited(upload, MAX_FRAME_BYTES, "A frame") for upload in frame_uploads]
    raw_frames = [frame for frame in raw_frames if frame]
    raw_video: bytes | None = None
    if not raw_frames:
        if video is None:
            raise HTTPException(status_code=400, detail="The uploaded frames were empty.")
        raw_video = await _read_limited(video, MAX_VIDEO_BYTES, "The video")
        if not raw_video:
            raise HTTPException(status_code=400, detail="The uploaded video was empty.")

    hint = " ".join(product_hint.split())[:MAX_HINT_CHARS]
    video_name = (video.filename or "") if video is not None else ""
    return StreamingResponse(
        _scan_stream(raw_frames, raw_video, video_name, hint),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@lru_cache(maxsize=64)
def _gzipped_glb(token: str, target: model_builder.Vec3 | None) -> bytes:
    """Gzip a built GLB once per unique URL (GLB geometry compresses by roughly half)."""
    return gzip.compress(model_builder.build_glb_from_token(token, target), compresslevel=6)


@app.api_route("/model.glb", methods=["GET", "HEAD"])
def model_glb(
    request: Request,
    r: str = Query(..., min_length=1, max_length=model_builder.MAX_TOKEN_CHARS, description="Recipe token from /scan."),
    w: float | None = Query(default=None, description="Target width in cm."),
    h: float | None = Query(default=None, description="Target height in cm."),
    d: float | None = Query(default=None, description="Target depth in cm."),
) -> Response:
    """Serve a generated model. Identical URLs always return identical bytes, so they are cached for a year."""
    try:
        target = model_builder.parse_target_dimensions(w, h, d)
        accepts_gzip = "gzip" in request.headers.get("accept-encoding", "").lower()
        body = _gzipped_glb(r, target) if accepts_gzip else model_builder.build_glb_from_token(r, target)
    except model_builder.RecipeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    headers = {
        "Cache-Control": "public, max-age=31536000, immutable",
        "Vary": "Accept-Encoding",
        "Content-Disposition": 'inline; filename="model.glb"',
    }
    if accepts_gzip:
        headers["Content-Encoding"] = "gzip"
    return Response(content=body, media_type="model/gltf-binary", headers=headers)
