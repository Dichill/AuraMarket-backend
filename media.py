"""
Image and video helpers for the scan pipeline.

* ``normalize_frame``   - decode any uploaded image, fix orientation, downscale, re-encode as JPEG.
* ``extract_frames``    - sample evenly spaced frames from a video with the bundled ffmpeg binary.
* ``crop_to_box``       - crop a frame to Gemini's 0-1000 normalized bounding box (with padding).
* ``to_jpeg_data_url``  - encode an image as a compact JPEG data URL for listing thumbnails.
"""

from __future__ import annotations

import base64
import io
import re
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageOps, UnidentifiedImageError

# Frames sent to Gemini: 1024 px on the long edge keeps detail while staying well under
# the 20 MB inline request limit even with 24 frames.
MAX_FRAME_EDGE_PX = 1024
FRAME_JPEG_QUALITY = 85

# Video container extensions ffmpeg is allowed to read (the upload's own name is never trusted).
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".3gp"}
FFMPEG_TIMEOUT_S = 120


class MediaError(ValueError):
    """Raised when an uploaded image or video cannot be decoded."""


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    """Encode a Pillow image as an optimized baseline JPEG."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """Convert any mode (RGBA, P, LA, CMYK...) to RGB, compositing transparency onto white."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def normalize_frame(data: bytes, max_edge: int = MAX_FRAME_EDGE_PX) -> bytes:
    """
    Decode an uploaded frame, apply EXIF rotation, shrink it to ``max_edge`` and return JPEG bytes.

    Raises:
        MediaError: when the bytes are not a readable image.
    """
    if not data:
        raise MediaError("An uploaded frame was empty.")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            # Phone photos store rotation in EXIF; bake it into the pixels before resizing.
            upright = ImageOps.exif_transpose(opened)
            rgb = _flatten_to_rgb(upright)
            rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            return _encode_jpeg(rgb, FRAME_JPEG_QUALITY)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise MediaError(f"Could not read an uploaded frame ({exc}).") from exc


def _probe_duration_seconds(ffmpeg: str, source: Path) -> float | None:
    """Read the container duration from ffmpeg's banner output (``Duration: HH:MM:SS.xx``)."""
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(source)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,  # ffmpeg exits non-zero because no output file is given; that is expected.
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr)
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return duration if duration > 0 else None


def extract_frames(video: bytes, filename: str, max_frames: int) -> list[bytes]:
    """
    Sample up to ``max_frames`` evenly spaced frames from a video.

    ffmpeg applies the phone's rotation metadata automatically, so portrait iPhone
    videos come out upright.

    Raises:
        MediaError: when the video cannot be decoded or yields no frames.
    """
    if not video:
        raise MediaError("The uploaded video was empty.")
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        suffix = ".mp4"

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix="aura_scan_") as workdir:
        source = Path(workdir) / f"input{suffix}"
        source.write_bytes(video)

        duration = _probe_duration_seconds(ffmpeg, source)
        # Spread the frames across the whole clip; unknown durations fall back to 2 frames/second.
        sample_fps = max_frames / duration if duration else 2.0
        pattern = Path(workdir) / "frame_%03d.jpg"
        try:
            subprocess.run(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source),
                    "-vf", f"fps={sample_fps:.5f}",
                    "-frames:v", str(max_frames),
                    "-q:v", "3",
                    str(pattern),
                ],
                capture_output=True,
                text=True,
                timeout=FFMPEG_TIMEOUT_S,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip().splitlines()[-1:] or ["unknown ffmpeg error"]
            raise MediaError(f"Could not decode the video ({detail[0]}).") from exc
        except subprocess.TimeoutExpired as exc:
            raise MediaError("Decoding the video took too long. Try a shorter clip.") from exc

        frame_paths = sorted(Path(workdir).glob("frame_*.jpg"))
        if not frame_paths:
            raise MediaError("No frames could be read from the video.")
        return [normalize_frame(path.read_bytes()) for path in frame_paths]


def crop_to_box(jpeg: bytes, box: list[int] | None, padding: float = 0.12) -> bytes:
    """
    Crop a frame to a ``[ymin, xmin, ymax, xmax]`` box normalized to 0-1000.

    A padding margin (fraction of the box size) keeps the item's edges in view. Invalid or
    tiny boxes return the full frame instead of failing.
    """
    with Image.open(io.BytesIO(jpeg)) as opened:
        image = _flatten_to_rgb(opened)
    width, height = image.size

    if box is not None and len(box) == 4:
        ymin, xmin, ymax, xmax = (max(0.0, min(1000.0, float(value))) for value in box)
        if ymax - ymin >= 20 and xmax - xmin >= 20:
            left, right = xmin / 1000 * width, xmax / 1000 * width
            top, bottom = ymin / 1000 * height, ymax / 1000 * height
            pad_x, pad_y = (right - left) * padding, (bottom - top) * padding
            crop_box = (
                int(max(0.0, left - pad_x)),
                int(max(0.0, top - pad_y)),
                int(min(float(width), right + pad_x)),
                int(min(float(height), bottom + pad_y)),
            )
            image = image.crop(crop_box)

    return _encode_jpeg(image, 92)


def to_jpeg_data_url(image_bytes: bytes, max_edge: int = 768, quality: int = 82) -> str:
    """
    Re-encode an image as a small JPEG ``data:`` URL (roughly 50-120 KB).

    The frontend stores this directly in the Firestore listing, which must stay under
    Firestore's 1 MiB document limit.

    Raises:
        MediaError: when the bytes are not a readable image.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = _flatten_to_rgb(opened)
    except (UnidentifiedImageError, OSError) as exc:
        raise MediaError(f"Could not read the generated image ({exc}).") from exc
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    encoded = base64.b64encode(_encode_jpeg(image, quality)).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
