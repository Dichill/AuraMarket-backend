"""
Gemini scan pipeline: video frames in, autofilled listing + 3D model + studio photo out.

Steps (progress is reported through ``emit`` so the frontend can show live status):

1. identify  - a vision model reads the frames: what the item is, its condition, a size
               estimate, the best frame and the item's bounding box.
2. research  - Interactions API + Google Search grounding: verified product name,
               official dimensions, MSRP and used prices, listing copy, sources.
3. photo     - Nano Banana turns the best frame into a white-background studio photo.
4. model     - a vision model decomposes the item into primitive parts, which
               ``model_builder`` validates and packs into a GLB recipe token.

Steps 2-4 run in parallel once step 1 has finished.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar
from urllib.parse import urlparse

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

import model_builder
from media import MediaError, crop_to_box, to_jpeg_data_url
from schemas import DimensionsCm, GeometryPlan, Identification, Research

logger = logging.getLogger("auramarket.pipeline")

Event = dict[str, object]
Emit = Callable[[Event], Awaitable[None]]
ParsedT = TypeVar("ParsedT")

IDENTIFY_TIMEOUT_S = 90.0
RESEARCH_TIMEOUT_S = 150.0
PHOTO_TIMEOUT_S = 90.0
GEOMETRY_TIMEOUT_S = 180.0
MAX_ERROR_CHARS = 280


class StepError(RuntimeError):
    """A pipeline step failed on every model in its fallback chain."""


def _env(name: str, default: str) -> str:
    """Read a non-empty environment variable or return ``default``."""
    value = os.getenv(name, "").strip()
    return value or default


@dataclass(frozen=True)
class PipelineConfig:
    """Gemini model names, overridable with environment variables on Cloud Run."""

    vision_model: str
    research_model: str
    geometry_model: str
    geometry_thinking: str
    image_model: str
    fallback_text_model: str
    fallback_image_model: str

    @classmethod
    def from_env(cls) -> PipelineConfig:
        """Build the configuration from environment variables with current defaults."""
        # Benchmarked on real scans: 3.8 Flash with medium thinking builds a 20+ part model in
        # ~11 s; 3.1 Pro is similar in quality but ~7x slower, so it is the fallback.
        return cls(
            vision_model=_env("GEMINI_VISION_MODEL", "gemini-3.8-flash"),
            research_model=_env("GEMINI_RESEARCH_MODEL", "gemini-3.8-flash"),
            geometry_model=_env("GEMINI_GEOMETRY_MODEL", "gemini-3.8-flash"),
            geometry_thinking=_env("GEMINI_GEOMETRY_THINKING", "medium"),
            image_model=_env("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image"),
            fallback_text_model=_env("GEMINI_FALLBACK_MODEL", "gemini-3.1-pro-preview"),
            fallback_image_model=_env("GEMINI_FALLBACK_IMAGE_MODEL", "gemini-3-pro-image"),
        )

    def as_dict(self) -> dict[str, str]:
        """Model names for status output."""
        return {
            "vision": self.vision_model,
            "research": self.research_model,
            "geometry": self.geometry_model,
            "image": self.image_model,
        }


@dataclass(frozen=True)
class ResearchOutcome:
    """Grounded research plus the Google Search metadata that must be shown with it."""

    research: Research
    search_queries: list[str]
    search_suggestions_html: str
    model: str


@dataclass(frozen=True)
class GeometryOutcome:
    """Validated primitive parts and their encoded recipe."""

    parts: list[model_builder.Part]
    dropped_parts: int
    recipe_token: str
    natural_size: model_builder.Vec3
    model: str


@dataclass(frozen=True)
class PhotoOutcome:
    """Listing thumbnail as a JPEG data URL, and whether Nano Banana produced it."""

    data_url: str
    source: str  # "studio" (Nano Banana) or "video_frame" (cropped best frame)
    model: str


def _progress(stage: str, status: str, message: str, detail: dict[str, object] | None = None) -> Event:
    """Build a progress event for the NDJSON stream."""
    event: Event = {"type": "progress", "stage": stage, "status": status, "message": message}
    if detail:
        event["detail"] = detail
    return event


def _describe_error(model: str, error: BaseException) -> str:
    """Short, user-readable description of why a model call failed."""
    if isinstance(error, genai_errors.APIError):
        text = f"{model}: {error.code} {error.status or ''} {error.message or ''}"
    elif isinstance(error, asyncio.TimeoutError):
        text = f"{model}: timed out"
    else:
        text = f"{model}: {type(error).__name__}: {error}"
    compact = " ".join(text.split())
    return compact if len(compact) <= MAX_ERROR_CHARS else f"{compact[:MAX_ERROR_CHARS]}..."


def _model_chain(*models: str) -> list[str]:
    """Deduplicate a primary + fallback model list while keeping its order."""
    return list(dict.fromkeys(model for model in models if model))


def _frame_parts(frames: list[bytes]) -> list[types.Part]:
    """Interleave 'Frame N:' labels with the images so Gemini can cite frame indices."""
    parts: list[types.Part] = []
    for index, frame in enumerate(frames):
        parts.append(types.Part.from_text(text=f"Frame {index}:"))
        parts.append(types.Part.from_bytes(data=frame, mime_type="image/jpeg"))
    return parts


def _finite_positive(value: float) -> bool:
    """True for real numbers greater than zero."""
    return math.isfinite(value) and value > 0


def _dims_tuple(dimensions: DimensionsCm) -> model_builder.Vec3 | None:
    """Convert schema dimensions to a clamped (w, h, d) tuple, or None when unusable."""
    values = (dimensions.width, dimensions.height, dimensions.depth)
    if not all(_finite_positive(value) for value in values):
        return None
    w, h, d = (round(min(max(value, 1.0), model_builder.MAX_EXTENT_CM), 1) for value in values)
    return (w, h, d)


def _match_dimensions(official: model_builder.Vec3, estimate: model_builder.Vec3) -> model_builder.Vec3 | None:
    """
    Align official dimensions with the video estimate.

    Spec sheets list sizes in inconsistent orders (W x D x H, H x W x D...), so this picks
    the axis order closest to what the video shows. Returns None when even the best order
    is off by more than 2.5x on some axis, which usually means research found a different
    product or variant.
    """
    best: model_builder.Vec3 | None = None
    best_score = math.inf
    for candidate in itertools.permutations(official):
        score = sum(abs(math.log(value / guess)) for value, guess in zip(candidate, estimate))
        if score < best_score:
            best_score, best = score, (candidate[0], candidate[1], candidate[2])
    if best is None:
        return None
    worst_ratio = max(max(value / guess, guess / value) for value, guess in zip(best, estimate))
    return best if worst_ratio <= 2.5 else None


def _clean_text(value: str, limit: int) -> str:
    """Collapse whitespace and cap the length of a model-written string."""
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1].rstrip()}…"


def _clean_paragraphs(value: str, limit: int) -> str:
    """Like ``_clean_text`` but keeps paragraph breaks."""
    paragraphs = [" ".join(block.split()) for block in value.replace("\r", "").split("\n")]
    joined = "\n\n".join(block for block in paragraphs if block)
    return joined if len(joined) <= limit else f"{joined[: limit - 1].rstrip()}…"


def _is_web_url(url: str) -> bool:
    """Accept only absolute http(s) URLs for the sources list."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


class ScanPipeline:
    """Runs the full scan for one upload. Safe to share across requests."""

    def __init__(self, client: genai.Client, config: PipelineConfig) -> None:
        self._client = client
        self._config = config

    @property
    def config(self) -> PipelineConfig:
        """The model configuration in use."""
        return self._config

    # ----------------------------------------------------------------------------------
    # Gemini helpers
    # ----------------------------------------------------------------------------------

    async def _generate_structured(
        self,
        *,
        step: str,
        models: list[str],
        contents: list[types.Part],
        schema: type[BaseModel],
        thinking_level: str,
        timeout_s: float,
        parse: Callable[[dict[str, object]], ParsedT],
    ) -> tuple[ParsedT, str]:
        """
        Call ``generate_content`` with a JSON schema, trying each model in order.

        ``parse`` validates the JSON; if it raises, the next model is tried as well.
        """
        failures: list[str] = []
        json_schema = schema.model_json_schema()
        for model in models:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=json_schema,
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(model=model, contents=contents, config=config),
                    timeout=timeout_s,
                )
                payload: object = json.loads(response.text or "")
                if not isinstance(payload, dict):
                    raise ValueError("the response was not a JSON object")
                return parse(payload), model
            except (genai_errors.APIError, asyncio.TimeoutError, json.JSONDecodeError, ValidationError, ValueError) as error:
                logger.warning("%s failed on %s: %s", step, model, error)
                failures.append(_describe_error(model, error))
        raise StepError(f"{step} failed. {' | '.join(failures)}")

    # ----------------------------------------------------------------------------------
    # Step 1: identification
    # ----------------------------------------------------------------------------------

    async def identify(self, frames: list[bytes], hint: str) -> tuple[Identification, str]:
        """Identify the item, its condition, size, best frame and bounding box."""
        hint_sentence = (
            f'The seller describes it as: "{hint}". Treat this as a strong hint, but trust what you can see.'
            if hint
            else "The seller did not describe the item."
        )
        focus_clause = " or the one matching the seller's description" if hint else ""
        prompt = "\n".join(
            [
                "You are the listing assistant for AuraMarket, a marketplace for pre-owned furniture, electronics, displays, art objects and fashion.",
                f"The images above are {len(frames)} frames (Frame 0 to Frame {len(frames) - 1}) sampled in order from a video a seller recorded of ONE item they want to sell.",
                hint_sentence,
                "",
                "Tasks:",
                "1. Identify the item as specifically as the frames allow: brand, model or product line, generation. Read visible logos, labels and text. If you are not sure, give a descriptive generic name and a low confidence instead of guessing a brand.",
                "2. Judge the condition only from visible evidence (scratches, stains, fading, dents, missing parts).",
                "3. Estimate the real-world width, height and depth in centimeters from visual cues (proportions, surrounding objects, standard sizes for this kind of product).",
                "4. Choose the frame that shows the whole item most clearly, ideally from a front three-quarter angle, and give the item's bounding box in that frame.",
                "5. Estimate a fair used-market price in USD.",
                f"If several objects are visible, the item is the one the video focuses on{focus_clause}.",
            ]
        )
        return await self._generate_structured(
            step="Identification",
            models=_model_chain(self._config.vision_model, self._config.fallback_text_model),
            contents=[*_frame_parts(frames), types.Part.from_text(text=prompt)],
            schema=Identification,
            thinking_level="low",
            timeout_s=IDENTIFY_TIMEOUT_S,
            parse=Identification.model_validate,
        )

    # ----------------------------------------------------------------------------------
    # Step 2: grounded research
    # ----------------------------------------------------------------------------------

    async def research(self, identification: Identification, hint: str) -> ResearchOutcome:
        """Research the item with Google Search grounding through the Interactions API."""
        estimate = identification.estimated_dimensions_cm
        prompt = "\n".join(
            [
                "Research a pre-owned item for a marketplace listing and autofill the listing fields.",
                "",
                "What the seller's video shows:",
                f"- Item: {identification.item_name}",
                f"- Brand: {identification.brand or 'unknown'} | Model: {identification.model or 'unknown'} | Type: {identification.item_type}",
                f"- Seller's description: {hint or 'none'}",
                f"- Colors and materials: {identification.colors_and_materials}",
                f"- Visible condition: {identification.condition}. {identification.condition_notes}",
                f"- Size estimated from the video: {estimate.width:.0f} x {estimate.height:.0f} x {estimate.depth:.0f} cm (width x height x depth)",
                f"- Suggested search: {identification.search_query}",
                "",
                "Use Google Search to:",
                "1. Confirm the exact product: official name, manufacturer and model number. If you cannot confirm it, use the closest match and set exact_match_confirmed to false.",
                "2. Find the official width, height and depth in centimeters (1 in = 2.54 cm), for the item as it appears in the video (for example including an attached stand). If there are no official dimensions, set official_dimensions_found to false and return the video estimate.",
                f"3. Find the original retail price (MSRP) and what this product sells for used in {identification.condition} condition (for example eBay sold listings, Facebook Marketplace, Chairish, 1stDibs, Swappa, Reverb, Back Market), then suggest a competitive listing price.",
                "4. Write the listing: a clear title of at most 80 characters, two short paragraphs of description, 3 to 6 selling points and up to 8 key specifications. Only state facts supported by the search results or visible in the video; never invent accessories, provenance or measurements.",
                "",
                "All prices are in USD. List the web pages you relied on in sources.",
            ]
        )
        schema = Research.model_json_schema()
        failures: list[str] = []
        for model in _model_chain(self._config.research_model, self._config.fallback_text_model):
            try:
                interaction = await asyncio.wait_for(
                    self._client.aio.interactions.create(
                        model=model,
                        input=prompt,
                        tools=[{"type": "google_search"}],
                        response_format={"type": "text", "mime_type": "application/json", "schema": schema},
                        store=False,
                    ),
                    timeout=RESEARCH_TIMEOUT_S,
                )
                output_text = getattr(interaction, "output_text", "")
                research = Research.model_validate_json(output_text if isinstance(output_text, str) else "")
                queries, suggestions = self._search_metadata(interaction.model_dump(mode="json", exclude_none=True))
                return ResearchOutcome(research=research, search_queries=queries, search_suggestions_html=suggestions, model=model)
            except (genai_errors.APIError, asyncio.TimeoutError, ValidationError, ValueError) as error:
                logger.warning("Research failed on %s: %s", model, error)
                failures.append(_describe_error(model, error))
        raise StepError(f"Research failed. {' | '.join(failures)}")

    @staticmethod
    def _search_metadata(dump: dict[str, object]) -> tuple[list[str], str]:
        """
        Pull the search queries and Google's "search suggestions" widget out of an interaction.

        Google's grounding terms require showing the suggestions HTML next to grounded results.
        """
        queries: list[str] = []
        suggestions = ""
        steps = dump.get("steps")
        if not isinstance(steps, list):
            return queries, suggestions
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("type") == "google_search_call":
                arguments = step.get("arguments")
                if isinstance(arguments, dict) and isinstance(arguments.get("queries"), list):
                    queries.extend(query for query in arguments["queries"] if isinstance(query, str))
            elif step.get("type") == "google_search_result" and not suggestions:
                results = step.get("result")
                if isinstance(results, list):
                    for result in results:
                        html = result.get("search_suggestions") if isinstance(result, dict) else None
                        if isinstance(html, str) and html.strip():
                            suggestions = html
                            break
        return list(dict.fromkeys(queries))[:8], suggestions

    # ----------------------------------------------------------------------------------
    # Step 3: studio photo
    # ----------------------------------------------------------------------------------

    async def studio_photo(self, frame: bytes, box: list[int], identification: Identification) -> PhotoOutcome:
        """Create the listing thumbnail with Nano Banana, falling back to the cropped frame."""
        crop = await asyncio.to_thread(crop_to_box, frame, box)
        prompt = "\n".join(
            [
                f"This is a frame from a seller's video of their {identification.item_name}. Turn it into a professional e-commerce product photo.",
                "- Keep the exact same item: identical shape, proportions, colors, materials, textures, logos and any visible wear. Do not redesign, restyle or improve it.",
                "- Remove everything else: room, floor, walls, people, hands, other objects and reflections of the room.",
                "- Show the item alone, centered and completely in frame with a small margin, on a seamless pure white background, with soft even studio lighting and a subtle natural contact shadow. Prefer a front three-quarter view.",
                "- If it has a screen, show the screen turned off (glossy black).",
                "- No text, labels, props or watermarks.",
            ]
        )
        failures: list[str] = []
        for model in _model_chain(self._config.image_model, self._config.fallback_image_model):
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1", image_size="1K"),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=model,
                        contents=[types.Part.from_bytes(data=crop, mime_type="image/jpeg"), types.Part.from_text(text=prompt)],
                        config=config,
                    ),
                    timeout=PHOTO_TIMEOUT_S,
                )
                image = self._first_image(response)
                if image is None:
                    raise ValueError("the model returned no image")
                data_url = await asyncio.to_thread(to_jpeg_data_url, image)
                return PhotoOutcome(data_url=data_url, source="studio", model=model)
            except (genai_errors.APIError, asyncio.TimeoutError, MediaError, ValueError) as error:
                logger.warning("Studio photo failed on %s: %s", model, error)
                failures.append(_describe_error(model, error))

        # Nano Banana unavailable: the real (cropped) video frame is still an honest thumbnail.
        data_url = await asyncio.to_thread(to_jpeg_data_url, crop)
        return PhotoOutcome(data_url=data_url, source="video_frame", model=f"none ({' | '.join(failures)})")

    @staticmethod
    def _first_image(response: types.GenerateContentResponse) -> bytes | None:
        """Return the first inline image in a response, if any."""
        for candidate in response.candidates or []:
            content = candidate.content
            if content is None:
                continue
            for part in content.parts or []:
                if part.inline_data is not None and part.inline_data.data:
                    return part.inline_data.data
        return None

    # ----------------------------------------------------------------------------------
    # Step 4: primitive 3D model
    # ----------------------------------------------------------------------------------

    async def geometry(self, frames: list[bytes], identification: Identification) -> GeometryOutcome:
        """Ask Gemini for a primitive-based model and validate it into a recipe token."""
        estimate = _dims_tuple(identification.estimated_dimensions_cm) or (60.0, 60.0, 60.0)
        width, height, depth = estimate
        prompt = "\n".join(
            [
                f"Build a simplified 3D model of the {identification.item_name} ({identification.item_type}) shown in the frames above, using basic shapes.",
                "It is shown in AR on a marketplace, so the silhouette, proportions, main components and colors must match the real item; fine details are not needed.",
                "",
                "Coordinate system (centimeters):",
                "- y is up and the item stands on the floor at y = 0.",
                "- The item's front faces +z (toward the viewer); +x is the viewer's right when looking at the front.",
                f"- The origin is the center of the item's footprint. The item is about {width:.0f} wide (x), {height:.0f} tall (y) and {depth:.0f} deep (z), so parts stay within x in [{-width / 2:.1f}, {width / 2:.1f}], y in [0, {height:.1f}], z in [{-depth / 2:.1f}, {depth / 2:.1f}].",
                "",
                "Parts:",
                "- shape: box, rounded_box, cylinder, cone, sphere, capsule, torus or lathe.",
                "- size: [x, y, z] full extents before rotation. Round shapes (cylinder, cone, capsule, torus, lathe) have their axis along y: size x and z are the diameters and size y is the length along the axis. A sphere with unequal sizes is an ellipsoid. torus: size x and z are the outer diameter, size y is the tube thickness.",
                "- pos: [x, y, z] center of the part's bounding box.",
                "- rot: [x, y, z] degrees about the part's center, applied x, then y, then z. A backrest leaning back is about [-12, 0, 0]; [0, 0, 90] lays a cylinder along x; [90, 0, 0] lays it along z.",
                "- color: '#rrggbb' sampled from the frames (the real, slightly shaded color; not pure white or black unless it truly is). metal, rough and alpha describe the surface (clear glass: alpha 0.3, rough 0.05; brushed metal: metal 1, rough 0.35; fabric: rough 0.9).",
                "- rounded_box: radius = corner radius in cm (cushions, upholstery, rounded plastic housings, speaker cabinets).",
                "- cone: top = top diameter / bottom diameter (0 = pointed tip, 0.6 = tapered leg or lampshade, above 1 = wider at the top).",
                "- lathe: profile = [[radius_cm, height_cm], ...] from the part's bottom (height 0) to its top, for turned or bulging round forms (turned legs, vases, bottles, lamp bases, knobs).",
                "",
                "Rules:",
                "- Typically use 15 to 45 parts (simple items may need fewer, never more than 60): every main component (for example seat, backrest, arms and each leg of a chair; screen, bezel, stand and base of a TV; body, neck, headstock and sound hole of a guitar) plus the characteristic design details that make this product recognizable (channel tufting, piping, buttons, turned legs, handles, knobs, grilles, logos as small colored shapes).",
                "- Parts must touch or slightly overlap so the model is one connected object, and anything resting on the floor reaches y = 0.",
                "- Flat round or oval forms (discs, round tabletops, clock faces, the bouts of a guitar body) are short cylinders or ellipsoids rotated [90, 0, 0] when they face the front; combine overlapping ones for curvy outlines.",
                "- Show screens turned off: a glossy near-black panel (color about #0b0d10, rough 0.1).",
                "- List repeated parts individually (four legs are four parts) and keep symmetric parts exactly mirrored.",
                "- Study every frame to understand the depth and the back of the item before placing parts.",
            ]
        )

        def parse(payload: dict[str, object]) -> tuple[list[model_builder.Part], int]:
            return model_builder.sanitize_plan(payload.get("parts"))

        (parts, dropped), model = await self._generate_structured(
            step="3D model",
            models=_model_chain(self._config.geometry_model, self._config.fallback_text_model),
            contents=[*_frame_parts(frames), types.Part.from_text(text=prompt)],
            schema=GeometryPlan,
            thinking_level=self._config.geometry_thinking,
            timeout_s=GEOMETRY_TIMEOUT_S,
            parse=parse,
        )
        token = model_builder.encode_recipe(parts)
        natural = await asyncio.to_thread(model_builder.natural_size_cm, parts)
        return GeometryOutcome(parts=parts, dropped_parts=dropped, recipe_token=token, natural_size=natural, model=model)

    # ----------------------------------------------------------------------------------
    # Orchestration
    # ----------------------------------------------------------------------------------

    async def run_scan(self, frames: list[bytes], hint: str, emit: Emit) -> None:
        """
        Run every step and emit progress, then exactly one ``result`` or ``error`` event.
        """
        timings: dict[str, int] = {}
        warnings: list[str] = []

        # Step 1 - identification (everything else depends on it).
        await emit(_progress("identify", "running", "Gemini is identifying the item in your video..."))
        started = time.monotonic()
        try:
            identification, vision_model = await self.identify(frames, hint)
        except StepError as error:
            await emit(_progress("identify", "failed", str(error)))
            await emit({"type": "error", "stage": "identify", "message": str(error)})
            return
        timings["identify"] = int((time.monotonic() - started) * 1000)
        if not identification.object_found:
            message = "Gemini could not find a single sellable item in this video. Film one item so it fills most of the frame and move slowly around it."
            await emit(_progress("identify", "failed", message))
            await emit({"type": "error", "stage": "identify", "message": message})
            return
        await emit(
            _progress(
                "identify",
                "done",
                f"Identified: {identification.item_name}",
                {
                    "itemName": identification.item_name,
                    "category": identification.category,
                    "confidence": round(min(max(identification.identification_confidence, 0.0), 1.0), 2),
                },
            )
        )

        best_index = min(max(identification.best_frame_index, 0), len(frames) - 1)

        async def run_research() -> ResearchOutcome | None:
            await emit(_progress("research", "running", "Researching specs and prices with Google Search..."))
            step_started = time.monotonic()
            try:
                outcome = await self.research(identification, hint)
            except StepError as error:
                warnings.append("Web research failed, so the price and description are Gemini's estimates from the video alone.")
                await emit(_progress("research", "failed", str(error)))
                return None
            timings["research"] = int((time.monotonic() - step_started) * 1000)
            await emit(_progress("research", "done", f"Found {outcome.research.verified_name}", {"sources": len(outcome.research.sources)}))
            return outcome

        async def run_photo() -> PhotoOutcome:
            await emit(_progress("photo", "running", "Creating a studio photo with Nano Banana..."))
            step_started = time.monotonic()
            outcome = await self.studio_photo(frames[best_index], identification.bounding_box, identification)
            timings["photo"] = int((time.monotonic() - step_started) * 1000)
            if outcome.source == "studio":
                await emit(_progress("photo", "done", "Studio photo ready"))
            else:
                warnings.append("Nano Banana could not create a studio photo, so the thumbnail is a cropped frame from your video.")
                await emit(_progress("photo", "failed", "Using a cropped video frame instead"))
            return outcome

        async def run_geometry() -> GeometryOutcome | None:
            await emit(_progress("model", "running", "Gemini is building the 3D model from your video..."))
            step_started = time.monotonic()
            try:
                outcome = await self.geometry(frames, identification)
            except (StepError, model_builder.RecipeError) as error:
                await emit(_progress("model", "failed", str(error)))
                return None
            timings["model"] = int((time.monotonic() - step_started) * 1000)
            await emit(_progress("model", "done", f"3D model built from {len(outcome.parts)} parts"))
            return outcome

        # Steps 2-4 in parallel.
        research_outcome, photo_outcome, geometry_outcome = await asyncio.gather(run_research(), run_photo(), run_geometry())
        if geometry_outcome is None:
            await emit(
                {
                    "type": "error",
                    "stage": "model",
                    "message": "Gemini could not build a 3D model from this video. Try again, or record a slower video that shows the item from several sides.",
                }
            )
            return
        if geometry_outcome.dropped_parts:
            warnings.append(f"{geometry_outcome.dropped_parts} malformed 3D part(s) were skipped.")

        await emit(_progress("finalize", "running", "Filling in your listing..."))
        result = self._assemble_result(identification, vision_model, research_outcome, photo_outcome, geometry_outcome, warnings, timings)
        await emit(_progress("finalize", "done", "Listing ready for review"))
        await emit(result)

    def _assemble_result(
        self,
        identification: Identification,
        vision_model: str,
        research_outcome: ResearchOutcome | None,
        photo: PhotoOutcome,
        geometry: GeometryOutcome,
        warnings: list[str],
        timings: dict[str, int],
    ) -> Event:
        """Merge every step into the listing payload the frontend autofills."""
        research = research_outcome.research if research_outcome else None

        # Dimensions: manufacturer data when it agrees with the video, else the video estimate.
        estimate = _dims_tuple(identification.estimated_dimensions_cm) or geometry.natural_size
        dimensions = estimate
        dimensions_source = "video_estimate"
        if research is not None and research.official_dimensions_found:
            official = _dims_tuple(research.official_dimensions_cm)
            matched = _match_dimensions(official, estimate) if official else None
            if matched is not None:
                dimensions, dimensions_source = matched, "manufacturer"
            else:
                warnings.append("The dimensions found online did not match the video, so the video estimate is used.")

        # Price: research suggestion first, then the vision estimate.
        suggested = research.suggested_price_usd if research and _finite_positive(research.suggested_price_usd) else 0.0
        fallback_price = identification.estimated_price_usd if _finite_positive(identification.estimated_price_usd) else 0.0
        price = float(round(min(suggested or fallback_price, 1_000_000.0)))
        if price <= 0:
            warnings.append("Gemini could not estimate a price; please enter one.")

        verified_name = _clean_text((research.verified_name if research else "") or identification.item_name, 160)
        title = _clean_text((research.headline if research else "") or verified_name, 120)
        description = _clean_paragraphs((research.description if research else "") or identification.summary, 3000)
        features = [_clean_text(feature, 140) for feature in (research.features if research else []) if feature.strip()][:6]
        specs = [
            {"label": _clean_text(spec.label, 60), "value": _clean_text(spec.value, 160)}
            for spec in (research.specs if research else [])
            if spec.label.strip() and spec.value.strip()
        ][:8]
        sources: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for source in research.sources if research else []:
            url = source.url.strip()
            if _is_web_url(url) and url not in seen_urls:
                seen_urls.add(url)
                sources.append({"title": _clean_text(source.title or urlparse(url).netloc, 120), "url": url})

        width, height, depth = dimensions
        return {
            "type": "result",
            "listing": {
                "title": title,
                "verifiedName": verified_name,
                "brand": _clean_text((research.manufacturer if research else "") or identification.brand, 80),
                "modelNumber": _clean_text((research.model_number if research else "") or identification.model, 80),
                "category": identification.category,
                "condition": identification.condition,
                "conditionNotes": _clean_text(identification.condition_notes, 300),
                "description": description,
                "features": features,
                "specs": specs,
                "price": price,
                "priceLow": float(round(research.used_price_low_usd)) if research and _finite_positive(research.used_price_low_usd) else None,
                "priceHigh": float(round(research.used_price_high_usd)) if research and _finite_positive(research.used_price_high_usd) else None,
                "msrp": float(round(research.msrp_usd)) if research and _finite_positive(research.msrp_usd) else None,
                "priceRationale": _clean_text(research.price_rationale, 300) if research else "",
                "dimensions": {"width": width, "height": height, "depth": depth, "unit": "cm"},
                "dimensionsSource": dimensions_source,
                "exactMatch": bool(research.exact_match_confirmed) if research else False,
                "identificationConfidence": round(min(max(identification.identification_confidence, 0.0), 1.0), 2),
            },
            "model": {
                "recipeToken": geometry.recipe_token,
                "partCount": len(geometry.parts),
                "naturalSize": {"width": geometry.natural_size[0], "height": geometry.natural_size[1], "depth": geometry.natural_size[2], "unit": "cm"},
                "modelPath": f"/model.glb?v={model_builder.RECIPE_VERSION}&w={width}&h={height}&d={depth}&r={geometry.recipe_token}",
            },
            "thumbnail": {"dataUrl": photo.data_url, "source": photo.source},
            "research": {
                "sources": sources,
                "searchQueries": research_outcome.search_queries if research_outcome else [],
                "searchSuggestionsHtml": research_outcome.search_suggestions_html if research_outcome else "",
            },
            "models": {
                "vision": vision_model,
                "research": research_outcome.model if research_outcome else "failed",
                "geometry": geometry.model,
                "image": photo.model,
            },
            "warnings": warnings,
            "timingsMs": timings,
        }
