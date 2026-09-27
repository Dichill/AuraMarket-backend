"""
Primitive-based 3D model builder.

Gemini cannot output mesh files, so the scan pipeline asks it to describe the item as a
list of simple parts (boxes, rounded boxes, cylinders, cones, ellipsoids, capsules, tori
and lathe / turned shapes), each with a real size, position, rotation and material.

This module:

1. sanitizes that plan (clamps sizes, angles, colors and material values so bad model
   output can never crash the build),
2. packs it into a compact, URL-safe *recipe token* (JSON -> zlib -> base64url), and
3. rebuilds a GLB from a token on demand, scaled to exact real-world dimensions.

Because ``GET /model.glb?r=<token>&w=..&h=..&d=..`` rebuilds the same GLB every time,
listings can store a plain HTTPS model URL without any file-storage bucket.

Coordinate conventions (identical to glTF): Y is up, the item's front faces +Z, +X is
the viewer's right when facing the front. Plans are in centimeters; GLBs are in meters.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import re
import zlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

# trimesh logs a full traceback each time it falls back from scipy sparse matrices to
# plain numpy. The fallback is correct, so keep the service logs readable.
logging.getLogger("trimesh").setLevel(logging.ERROR)

RECIPE_VERSION = 1
MAX_PARTS = 64
MAX_PROFILE_POINTS = 24
MAX_TOKEN_CHARS = 16_000
MAX_RECIPE_BYTES = 128 * 1024
MIN_PART_CM = 0.2
MAX_EXTENT_CM = 3_000.0
MIN_TARGET_CM = 1.0
ROUND_SEGMENTS = 40
ROUNDED_BOX_SEGMENTS = 4
SMOOTH_ANGLE_RAD = math.radians(35)
DEFAULT_COLOR = "9e9e9e"

SHAPE_CODES: dict[str, str] = {
    "box": "b",
    "rounded_box": "rb",
    "cylinder": "cy",
    "cone": "co",
    "sphere": "sp",
    "capsule": "ca",
    "torus": "to",
    "lathe": "la",
}
CODE_TO_SHAPE: dict[str, str] = {code: shape for shape, code in SHAPE_CODES.items()}
# Names models sometimes use instead of the schema's enum values.
SHAPE_ALIASES: dict[str, str] = {
    "cube": "box",
    "cuboid": "box",
    "roundedbox": "rounded_box",
    "ellipsoid": "sphere",
    "ball": "sphere",
    "rod": "cylinder",
    "tube": "cylinder",
    "frustum": "cone",
    "ring": "torus",
    "revolve": "lathe",
}
HEX_COLOR = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# trimesh builds round primitives along +Z; rotate them so their axis is glTF's +Y.
_Z_TO_Y = trimesh.transformations.rotation_matrix(-math.pi / 2, [1.0, 0.0, 0.0])

Vec3 = tuple[float, float, float]
ProfilePoints = tuple[tuple[float, float], ...]


class RecipeError(ValueError):
    """Raised when a geometry plan or recipe token is invalid."""


@dataclass(frozen=True)
class Part:
    """One validated primitive. Lengths are centimeters, angles are degrees."""

    shape: str
    size: Vec3
    pos: Vec3
    rot: Vec3
    color: str  # six lowercase hex digits, sRGB, no leading '#'
    metal: float
    rough: float
    alpha: float
    radius: float  # rounded_box corner radius
    top: float  # cone top-to-bottom diameter ratio
    profile: ProfilePoints  # lathe outline as (radius, height) pairs


# --------------------------------------------------------------------------------------
# Sanitizing Gemini output
# --------------------------------------------------------------------------------------


def _as_float(value: object) -> float | None:
    """Return ``value`` as a finite float, or None when it is not a usable number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _float_or(value: object, default: float) -> float:
    """Like ``_as_float`` but falls back to ``default``."""
    number = _as_float(value)
    return default if number is None else number


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


def _vec3(value: object) -> Vec3 | None:
    """Parse a ``[x, y, z]`` list of numbers."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    x, y, z = (_as_float(item) for item in value)
    if x is None or y is None or z is None:
        return None
    return (x, y, z)


def _color(value: object) -> str:
    """Parse '#rrggbb' / 'rrggbb' / '#rgb' into six lowercase hex digits."""
    if isinstance(value, str):
        match = HEX_COLOR.match(value.strip())
        if match is not None:
            digits = match.group(1).lower()
            if len(digits) == 3:
                digits = "".join(character * 2 for character in digits)
            return digits
    return DEFAULT_COLOR


def _profile(value: object) -> ProfilePoints:
    """Parse a lathe outline; returns an empty tuple when it cannot form a solid."""
    if not isinstance(value, (list, tuple)):
        return ()
    points: list[tuple[float, float]] = []
    for item in value:
        radius: float | None = None
        height: float | None = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            radius, height = _as_float(item[0]), _as_float(item[1])
        elif isinstance(item, dict):
            # Tolerate object-style points such as {"radius_cm": 3, "y_cm": 10}.
            radius = _as_float(item.get("radius_cm", item.get("radius", item.get("r"))))
            height = _as_float(item.get("y_cm", item.get("height_cm", item.get("y"))))
        if radius is None or height is None:
            continue
        point = (round(_clamp(abs(radius), 0.0, MAX_EXTENT_CM), 1), round(_clamp(height, -MAX_EXTENT_CM, MAX_EXTENT_CM), 1))
        # Consecutive duplicates would create zero-area faces.
        if not points or points[-1] != point:
            points.append(point)

    if len(points) > MAX_PROFILE_POINTS:
        keep = sorted({int(round(index)) for index in np.linspace(0, len(points) - 1, MAX_PROFILE_POINTS)})
        points = [points[index] for index in keep]
    if len(points) < 2:
        return ()
    heights = [height for _, height in points]
    if max(heights) - min(heights) < MIN_PART_CM or max(radius for radius, _ in points) < MIN_PART_CM / 2:
        return ()
    return tuple(points)


def sanitize_part(raw: object) -> Part | None:
    """
    Validate one part from Gemini's plan (or a decoded recipe row).

    Returns None when the part is unusable (unknown shape, missing size or position).
    Lathe parts without a usable outline degrade to cylinders instead of being dropped.
    """
    if not isinstance(raw, dict):
        return None

    shape_value = raw.get("shape")
    shape = shape_value.strip().lower().replace(" ", "_").replace("-", "_") if isinstance(shape_value, str) else ""
    shape = SHAPE_ALIASES.get(shape, shape)
    if shape not in SHAPE_CODES:
        return None

    profile = _profile(raw.get("profile")) if shape == "lathe" else ()
    if shape == "lathe" and not profile:
        shape = "cylinder"

    size = _vec3(raw.get("size"))
    if size is None and profile:
        # Derive the bounding box from the outline itself.
        diameter = 2 * max(radius for radius, _ in profile)
        span = max(height for _, height in profile) - min(height for _, height in profile)
        size = (diameter, span, diameter)
    position = _vec3(raw.get("pos"))
    if size is None or position is None:
        return None
    rotation = _vec3(raw.get("rot")) or (0.0, 0.0, 0.0)

    clean_size: Vec3 = (
        round(_clamp(abs(size[0]), MIN_PART_CM, MAX_EXTENT_CM), 1),
        round(_clamp(abs(size[1]), MIN_PART_CM, MAX_EXTENT_CM), 1),
        round(_clamp(abs(size[2]), MIN_PART_CM, MAX_EXTENT_CM), 1),
    )
    clean_position: Vec3 = (
        round(_clamp(position[0], -MAX_EXTENT_CM, MAX_EXTENT_CM), 1),
        round(_clamp(position[1], -MAX_EXTENT_CM, MAX_EXTENT_CM), 1),
        round(_clamp(position[2], -MAX_EXTENT_CM, MAX_EXTENT_CM), 1),
    )
    # Wrap angles into [-180, 180) so equivalent rotations encode identically.
    clean_rotation: Vec3 = (
        round(((rotation[0] + 180.0) % 360.0) - 180.0, 1),
        round(((rotation[1] + 180.0) % 360.0) - 180.0, 1),
        round(((rotation[2] + 180.0) % 360.0) - 180.0, 1),
    )

    corner_radius = 0.0
    if shape == "rounded_box":
        corner_radius = round(_clamp(_float_or(raw.get("radius"), 0.0), 0.0, min(clean_size) / 2), 1)
    top_ratio = round(_clamp(_float_or(raw.get("top"), 0.0), 0.0, 4.0), 2) if shape == "cone" else 0.0

    return Part(
        shape=shape,
        size=clean_size,
        pos=clean_position,
        rot=clean_rotation,
        color=_color(raw.get("color")),
        metal=round(_clamp(_float_or(raw.get("metal"), 0.0), 0.0, 1.0), 2),
        rough=round(_clamp(_float_or(raw.get("rough"), 0.6), 0.0, 1.0), 2),
        alpha=round(_clamp(_float_or(raw.get("alpha"), 1.0), 0.05, 1.0), 2),
        radius=corner_radius,
        top=top_ratio,
        profile=profile,
    )


def sanitize_plan(raw_parts: object) -> tuple[list[Part], int]:
    """
    Validate every part of a plan.

    Returns:
        The usable parts (at most ``MAX_PARTS``) and how many raw parts were dropped.

    Raises:
        RecipeError: when no usable part remains.
    """
    if not isinstance(raw_parts, list):
        raise RecipeError("The geometry plan did not contain a list of parts.")
    parts = [part for part in (sanitize_part(item) for item in raw_parts) if part is not None]
    dropped = len(raw_parts) - len(parts)
    if len(parts) > MAX_PARTS:
        dropped += len(parts) - MAX_PARTS
        parts = parts[:MAX_PARTS]
    if not parts:
        raise RecipeError("None of the generated 3D parts were usable.")
    return parts, dropped


# --------------------------------------------------------------------------------------
# Recipe token encoding
# --------------------------------------------------------------------------------------


def _part_to_row(part: Part) -> list[object]:
    """Pack a part into a compact positional row for the recipe JSON."""
    extra: object = 0
    if part.shape == "rounded_box":
        extra = part.radius
    elif part.shape == "cone":
        extra = part.top
    elif part.shape == "lathe":
        extra = [[radius, height] for radius, height in part.profile]
    return [
        SHAPE_CODES[part.shape],
        *part.size,
        *part.pos,
        *part.rot,
        part.color,
        part.metal,
        part.rough,
        part.alpha,
        extra,
    ]


def _row_to_part(row: object) -> Part | None:
    """Unpack a recipe row, re-running full validation so crafted tokens stay safe."""
    if not isinstance(row, list) or len(row) != 15 or not isinstance(row[0], str):
        return None
    shape = CODE_TO_SHAPE.get(row[0])
    if shape is None:
        return None
    extra = row[14]
    return sanitize_part(
        {
            "shape": shape,
            "size": row[1:4],
            "pos": row[4:7],
            "rot": row[7:10],
            "color": row[10],
            "metal": row[11],
            "rough": row[12],
            "alpha": row[13],
            "radius": extra if shape == "rounded_box" else 0,
            "top": extra if shape == "cone" else 0,
            "profile": extra if shape == "lathe" else [],
        }
    )


def encode_recipe(parts: list[Part]) -> str:
    """Serialize parts into a URL-safe token (compact JSON -> zlib -> base64url, no padding)."""
    payload = {"v": RECIPE_VERSION, "p": [_part_to_row(part) for part in parts]}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    token = base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode("ascii").rstrip("=")
    if len(token) > MAX_TOKEN_CHARS:
        raise RecipeError("The 3D model is too detailed to encode; try again for a simpler plan.")
    return token


def decode_recipe(token: str) -> list[Part]:
    """
    Parse a recipe token back into validated parts.

    Raises:
        RecipeError: for malformed, oversized or unsupported tokens.
    """
    if not token or len(token) > MAX_TOKEN_CHARS or TOKEN_PATTERN.match(token) is None:
        raise RecipeError("Invalid model recipe.")
    try:
        packed = base64.urlsafe_b64decode(f"{token}{'=' * (-len(token) % 4)}")
        inflater = zlib.decompressobj()
        # max_length caps the output so a tiny token cannot expand into a huge payload.
        raw = inflater.decompress(packed, MAX_RECIPE_BYTES)
        if inflater.unconsumed_tail:
            raise RecipeError("Model recipe is too large.")
        payload: object = json.loads(raw.decode("utf-8"))
    except (binascii.Error, zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecipeError("Invalid model recipe.") from exc

    if not isinstance(payload, dict) or payload.get("v") != RECIPE_VERSION:
        raise RecipeError("Unsupported model recipe version.")
    rows = payload.get("p")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_PARTS:
        raise RecipeError("Model recipe has an invalid number of parts.")
    parts = [part for part in (_row_to_part(row) for row in rows) if part is not None]
    if not parts:
        raise RecipeError("Model recipe has no valid parts.")
    return parts


# --------------------------------------------------------------------------------------
# Mesh construction
# --------------------------------------------------------------------------------------


def _fit_to_size(mesh: trimesh.Trimesh, size: Vec3) -> trimesh.Trimesh:
    """Center the mesh's bounding box on the origin and stretch it to exactly ``size``."""
    lower, upper = mesh.bounds
    mesh.apply_translation(-(lower + upper) / 2.0)
    extents = mesh.extents
    safe_extents = np.where(extents > 1e-9, extents, 1.0)
    factors = np.asarray(size, dtype=np.float64) / safe_extents
    mesh.apply_transform(np.diag([factors[0], factors[1], factors[2], 1.0]))
    return mesh


def _revolve(points: list[tuple[float, float]]) -> trimesh.Trimesh:
    """Revolve a (radius, height) outline around the vertical axis into a closed solid."""
    closed = list(points)
    # Close the outline onto the axis at both ends so the result is watertight.
    if closed[0][0] > 0:
        closed.insert(0, (0.0, closed[0][1]))
    if closed[-1][0] > 0:
        closed.append((0.0, closed[-1][1]))
    mesh = trimesh.creation.revolve(linestring=np.asarray(closed, dtype=np.float64), sections=ROUND_SEGMENTS)
    if mesh.volume < 0:
        # A top-to-bottom outline produces inward-facing normals.
        mesh.invert()
    mesh.apply_transform(_Z_TO_Y)
    return mesh


def _rounded_box(size: Vec3, radius: float) -> trimesh.Trimesh:
    """
    Build a box with rounded edges and corners.

    Each cube face is a grid (dense near the edges) whose vertices are projected onto the
    rounded surface: ``inner_box_point + direction * radius``. No convex-hull library is
    needed and flat faces stay perfectly flat.
    """
    half = np.asarray(size, dtype=np.float64) / 2.0
    corner = min(radius, float(half.min())) * 0.999
    if corner < 0.05:
        return trimesh.creation.box(extents=size)
    inner = half - corner
    # tan() spacing gives evenly spaced arc segments after projection (45 degrees per face).
    arc_offsets = corner * np.tan(np.linspace(0.0, math.pi / 4.0, ROUNDED_BOX_SEGMENTS + 1))

    def axis_samples(inner_half: float) -> np.ndarray:
        positive = inner_half + arc_offsets
        if inner_half > 1e-6:
            return np.concatenate([-positive[::-1], positive])
        return np.concatenate([-positive[::-1], positive[1:]])  # avoid a duplicate sample at 0

    samples = [axis_samples(float(inner[axis])) for axis in range(3)]
    vertex_blocks: list[np.ndarray] = []
    face_blocks: list[np.ndarray] = []
    offset = 0
    for axis in range(3):
        # (u, v) are chosen so that u x v points along +axis (right-handed).
        u_axis, v_axis = (axis + 1) % 3, (axis + 2) % 3
        grid_u, grid_v = np.meshgrid(samples[u_axis], samples[v_axis], indexing="ij")
        rows, cols = grid_u.shape
        index = np.arange(rows * cols).reshape(rows, cols)
        a, b = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
        c, d = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
        triangles = np.concatenate([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)])
        for sign in (1.0, -1.0):
            points = np.zeros((rows * cols, 3), dtype=np.float64)
            points[:, axis] = sign * half[axis]
            points[:, u_axis] = grid_u.ravel()
            points[:, v_axis] = grid_v.ravel()
            vertex_blocks.append(points)
            # Reverse the winding on the negative face so every normal points outward.
            face_blocks.append((triangles if sign > 0 else triangles[:, ::-1]) + offset)
            offset += len(points)

    vertices = np.concatenate(vertex_blocks)
    clamped = np.clip(vertices, -inner, inner)
    delta = vertices - clamped
    length = np.linalg.norm(delta, axis=1, keepdims=True)
    direction = np.divide(delta, length, out=np.zeros_like(delta), where=length > 1e-12)
    # process=True merges the duplicated seam vertices shared by neighbouring faces.
    return trimesh.Trimesh(vertices=clamped + direction * corner, faces=np.concatenate(face_blocks), process=True)


def _base_mesh(part: Part) -> trimesh.Trimesh:
    """Create the part's shape centered on the origin with bounding box == ``part.size``."""
    width, height, depth = part.size
    if part.shape == "box":
        return trimesh.creation.box(extents=part.size)
    if part.shape == "rounded_box":
        return _rounded_box(part.size, part.radius)
    if part.shape == "sphere":
        # Small spheres (buttons, knobs) do not need the extra triangles of large ones.
        subdivisions = 3 if max(part.size) >= 15 else 2
        return _fit_to_size(trimesh.creation.icosphere(subdivisions=subdivisions, radius=0.5), part.size)
    if part.shape == "cylinder":
        mesh = trimesh.creation.cylinder(radius=0.5, height=1.0, sections=ROUND_SEGMENTS)
        mesh.apply_transform(_Z_TO_Y)
        return _fit_to_size(mesh, part.size)
    if part.shape == "cone":
        if part.top <= 0.02:
            mesh = trimesh.creation.cone(radius=0.5, height=1.0, sections=ROUND_SEGMENTS)
            mesh.apply_transform(_Z_TO_Y)
        else:
            mesh = _revolve([(0.5, 0.0), (0.5 * part.top, 1.0)])
        return _fit_to_size(mesh, part.size)
    if part.shape == "capsule":
        diameter = min(width, depth)
        straight = max(height - diameter, 0.01)
        # count = (latitude, longitude) sections; trimesh's default (32x32) is ~4x heavier.
        mesh = trimesh.creation.capsule(height=straight, radius=diameter / 2, count=[24, 12])
        mesh.apply_transform(_Z_TO_Y)
        return _fit_to_size(mesh, part.size)
    if part.shape == "torus":
        tube = height / 2
        outer = max(width, depth) / 2
        # Keep a visible hole even when the model asks for an almost-solid ring.
        major = max(outer - tube, tube * 1.05)
        mesh = trimesh.creation.torus(major_radius=major, minor_radius=tube, major_sections=40, minor_sections=12)
        mesh.apply_transform(_Z_TO_Y)
        return _fit_to_size(mesh, part.size)
    if part.shape == "lathe":
        return _fit_to_size(_revolve(list(part.profile)), part.size)
    raise RecipeError(f"Unsupported shape '{part.shape}'.")


def _part_mesh(part: Part) -> trimesh.Trimesh:
    """Build a part and move it into place (rotation about its center, then translation)."""
    mesh = _base_mesh(part)
    rx, ry, rz = (math.radians(angle) for angle in part.rot)
    # 'sxyz' = rotate about the fixed X axis, then Y, then Z (matches the prompt's wording).
    mesh.apply_transform(trimesh.transformations.euler_matrix(rx, ry, rz, axes="sxyz"))
    mesh.apply_translation(part.pos)
    return mesh


def _srgb_to_linear(channel: float) -> float:
    """Convert one sRGB channel (0-1) to linear light; glTF color factors are linear."""
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _material(part: Part, index: int) -> PBRMaterial:
    """Create the part's PBR material from its sRGB hex color and surface values."""
    srgb = [int(part.color[offset:offset + 2], 16) / 255 for offset in (0, 2, 4)]
    linear = [int(round(_srgb_to_linear(channel) * 255)) for channel in srgb]
    transparent = part.alpha < 0.98
    return PBRMaterial(
        name=f"mat_{index:02d}",
        baseColorFactor=[*linear, int(round(part.alpha * 255))],
        metallicFactor=part.metal,
        roughnessFactor=part.rough,
        alphaMode="BLEND" if transparent else "OPAQUE",
        doubleSided=transparent,
    )


def _placed_meshes(parts: list[Part]) -> list[trimesh.Trimesh]:
    """Build every part in plan coordinates (centimeters)."""
    return [_part_mesh(part) for part in parts]


def natural_size_cm(parts: list[Part]) -> Vec3:
    """Overall width, height and depth of the plan as Gemini drew it, in centimeters."""
    meshes = _placed_meshes(parts)
    lower = np.min([mesh.bounds[0] for mesh in meshes], axis=0)
    upper = np.max([mesh.bounds[1] for mesh in meshes], axis=0)
    extent = upper - lower
    return (round(float(extent[0]), 1), round(float(extent[1]), 1), round(float(extent[2]), 1))


def build_glb(parts: list[Part], target_cm: Vec3 | None) -> bytes:
    """
    Build the GLB for a plan.

    The whole model is centered on X/Z, placed on the floor (lowest point at y = 0),
    stretched per axis so its bounding box equals ``target_cm`` exactly (when given),
    and converted from centimeters to meters.
    """
    meshes = _placed_meshes(parts)
    lower = np.min([mesh.bounds[0] for mesh in meshes], axis=0)
    upper = np.max([mesh.bounds[1] for mesh in meshes], axis=0)
    natural = np.maximum(upper - lower, 1e-6)
    anchor = np.array([(lower[0] + upper[0]) / 2.0, lower[1], (lower[2] + upper[2]) / 2.0])
    scale = np.ones(3) if target_cm is None else np.asarray(target_cm, dtype=np.float64) / natural

    placement = np.diag([*(scale * 0.01), 1.0]) @ trimesh.transformations.translation_matrix(-anchor)
    scene = trimesh.Scene()
    for index, (part, mesh) in enumerate(zip(parts, meshes)):
        mesh.apply_transform(placement)
        # Split vertices along sharp edges only: flat faces stay crisp, curves look smooth.
        shaded = trimesh.graph.smooth_shade(mesh, angle=SMOOTH_ANGLE_RAD)
        shaded.visual = TextureVisuals(material=_material(part, index))
        scene.add_geometry(shaded, node_name=f"part_{index:02d}_{part.shape}", geom_name=f"part_{index:02d}")

    exported = scene.export(file_type="glb", include_normals=True)
    if not isinstance(exported, bytes):
        raise RecipeError("GLB export failed.")
    return exported


def parse_target_dimensions(width: float | None, height: float | None, depth: float | None) -> Vec3 | None:
    """
    Validate optional target dimensions from a request.

    Raises:
        RecipeError: when only some dimensions are given or a value is not a finite number.
    """
    if width is None or height is None or depth is None:
        if width is None and height is None and depth is None:
            return None
        raise RecipeError("Provide all three dimensions (w, h, d) in centimeters, or none.")
    if not all(math.isfinite(value) for value in (width, height, depth)):
        raise RecipeError("Dimensions must be finite numbers.")
    return (
        round(_clamp(width, MIN_TARGET_CM, MAX_EXTENT_CM), 1),
        round(_clamp(height, MIN_TARGET_CM, MAX_EXTENT_CM), 1),
        round(_clamp(depth, MIN_TARGET_CM, MAX_EXTENT_CM), 1),
    )


@lru_cache(maxsize=64)
def build_glb_from_token(token: str, target_cm: Vec3 | None) -> bytes:
    """Decode a recipe token and build its GLB (cached: identical URLs are rebuilt once)."""
    return build_glb(decode_recipe(token), target_cm)
