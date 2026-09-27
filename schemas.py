"""
Structured-output schemas for the AuraMarket scan pipeline.

Every model in this file is sent to Gemini as a JSON schema (via
``Model.model_json_schema()``) so the model is forced to answer in exactly this
shape. Numeric ranges are described in the field descriptions instead of being
enforced with pydantic constraints: Gemini treats them as guidance, and the
pipeline clamps values itself so one slightly out-of-range number never throws
away an otherwise good answer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Enumerations shared with the frontend (see src/types/index.ts in AuraMarket-Frontend).
Category = Literal["Furniture", "Audio & Tech", "Displays", "Art & Objects", "Fashion"]
Condition = Literal["Mint / Unopened", "Like New", "Excellent", "Good", "Fair"]
ShapeKind = Literal["box", "rounded_box", "cylinder", "cone", "sphere", "capsule", "torus", "lathe"]


class DimensionsCm(BaseModel):
    """Real-world bounding box of an item, measured while facing its front."""

    width: float = Field(description="Left-to-right size in centimeters.")
    height: float = Field(description="Floor-to-top size in centimeters.")
    depth: float = Field(description="Front-to-back size in centimeters.")


class Identification(BaseModel):
    """What Gemini sees in the uploaded video frames (step 1 of the scan)."""

    object_found: bool = Field(description="False when no single sellable physical item is clearly visible.")
    item_name: str = Field(description="Most specific product name the frames support, e.g. 'Herman Miller Eames Lounge Chair'. Use a descriptive generic name when the exact product is unknown.")
    brand: str = Field(description="Brand or maker, or an empty string when it cannot be determined.")
    model: str = Field(description="Model name or number, or an empty string when it cannot be determined.")
    item_type: str = Field(description="Generic item type, e.g. 'tufted accent chair' or '55-inch LED TV'.")
    identification_confidence: float = Field(description="0 to 1 confidence that item_name is the exact product.")
    category: Category
    condition: Condition
    condition_notes: str = Field(description="Visible wear, damage or signs of use. Write 'No visible wear' when there is none.")
    colors_and_materials: str = Field(description="Main colors and materials, e.g. 'cream velvet upholstery, white lacquered wood legs'.")
    estimated_dimensions_cm: DimensionsCm
    estimated_price_usd: float = Field(description="Rough used-market price in USD for this item in this condition.")
    summary: str = Field(description="Two factual sentences describing the item as it appears in the frames.")
    best_frame_index: int = Field(description="Index of the frame that shows the whole item most clearly, ideally from a front three-quarter angle.")
    bounding_box: list[int] = Field(description="[ymin, xmin, ymax, xmax] of the whole item inside the best frame, normalized to 0-1000.")
    search_query: str = Field(description="The web search query most likely to find this exact product's specifications and prices.")


class SpecEntry(BaseModel):
    """One row of the product specification table."""

    label: str = Field(description="Specification name, e.g. 'Material' or 'Screen size'.")
    value: str = Field(description="Specification value, e.g. 'Walnut veneer' or '55 in'.")


class SourceLink(BaseModel):
    """A web page the research step relied on."""

    title: str = Field(description="Page or site title.")
    url: str = Field(description="Full https URL of the page.")


class Research(BaseModel):
    """Grounded (Google Search) research used to autofill the listing (step 2)."""

    verified_name: str = Field(description="Official full product name as the manufacturer or retailers list it.")
    manufacturer: str = Field(description="Manufacturer, or an empty string when unknown.")
    model_number: str = Field(description="Model number or SKU, or an empty string when unknown.")
    exact_match_confirmed: bool = Field(description="True only when search results confirm this exact product.")
    official_dimensions_found: bool = Field(description="True when official width, height and depth were found in the search results.")
    official_dimensions_cm: DimensionsCm
    msrp_usd: float = Field(description="Original retail price in USD, or 0 when unknown.")
    used_price_low_usd: float = Field(description="Low end of current used-market prices in USD for this condition.")
    used_price_high_usd: float = Field(description="High end of current used-market prices in USD for this condition.")
    suggested_price_usd: float = Field(description="Recommended listing price in USD for this item in this condition.")
    price_rationale: str = Field(description="One sentence explaining the suggested price.")
    headline: str = Field(description="Listing title, at most 80 characters, no emojis.")
    description: str = Field(description="Two short paragraphs for the listing: what it is and why it is desirable, then condition and practical details.")
    features: list[str] = Field(description="Three to six short selling points.")
    specs: list[SpecEntry] = Field(description="Up to eight key specifications.")
    sources: list[SourceLink] = Field(description="Up to six web pages the answer relied on.")


class GeometryPart(BaseModel):
    """One primitive shape of the simplified 3D model."""

    name: str = Field(description="Short part name, e.g. 'front left leg'.")
    shape: ShapeKind
    size: list[float] = Field(description="[x, y, z] full extents in cm before rotation. Round shapes (cylinder, cone, capsule, torus, lathe) have their axis along y, so x and z are diameters.")
    pos: list[float] = Field(description="[x, y, z] center of the part in cm (y = 0 is the floor).")
    rot: list[float] = Field(description="[x, y, z] rotation in degrees about the part's own center, applied x, then y, then z. Use [0, 0, 0] when upright.")
    color: str = Field(description="Hex color '#rrggbb' sampled from the frames.")
    metal: float = Field(description="Metalness 0 to 1 (1 = bare metal).")
    rough: float = Field(description="Roughness 0 to 1 (0 = mirror, 1 = matte fabric).")
    alpha: float = Field(description="Opacity 0.05 to 1 (1 = opaque, about 0.3 for clear glass).")
    radius: float = Field(default=0.0, description="rounded_box only: corner radius in cm.")
    top: float = Field(default=0.0, description="cone only: top diameter divided by bottom diameter (0 = pointed tip, 0.7 = gently tapered).")
    profile: list[list[float]] = Field(default_factory=list, description="lathe only: [[radius_cm, height_cm], ...] outline of a round object from its bottom (height 0) to its top.")


class GeometryPlan(BaseModel):
    """Gemini's primitive-based reconstruction of the item (step 3)."""

    overall_size_cm: DimensionsCm
    parts: list[GeometryPart] = Field(description="8 to 60 parts that together form the item.")
