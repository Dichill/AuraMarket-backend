import os
import time
import json
import tempfile
from typing import List, Dict, Any
from google import genai
from google.genai import types
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load local .env file if it exists
load_dotenv()

app = FastAPI(title="AuraMarket - Hackathon Backend")

# CRITICAL: Allow React frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini Client
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing!")
client = genai.Client(api_key=api_key)

# ==========================================
# SCHEMAS FOR /analyze (Template Scaling)
# ==========================================
class Identification(BaseModel):
    verified_name: str
    brand: str

class SpatialData(BaseModel):
    width_cm: float
    height_cm: float
    depth_cm: float
    anchors: str

class MarketAnalysis(BaseModel):
    condition: str
    reasoning: str
    price_usd: float

class ScaleVector(BaseModel):
    x: float
    y: float
    z: float

class Rendering(BaseModel):
    template_id: str
    scale_vector: ScaleVector

class Marketing(BaseModel):
    headline: str
    features: List[str]

class DigitalTwin(BaseModel):
    identification: Identification
    spatial_data: SpatialData
    market_analysis: MarketAnalysis
    marketing: Marketing
    rendering: Rendering

# ==========================================
# SCHEMAS FOR /extract-3d (Hybrid Approach)
# ==========================================
class Extracted3DModel(BaseModel):
    """Response for the hybrid 3D extraction."""
    object_name: str
    description: str
    obj_file_content: str
    recommended_texture_prompt: str

# ==========================================
# PROMPTS
# ==========================================
SYSTEM_PROMPT_ANALYZE = """
You are the core Spatial Reasoning and Market Valuation Engine for an AR e-commerce platform. Your objective is to ingest raw video and output a structured "Digital Twin" dataset.

### DIRECTIVES:
1. SPATIAL MEASUREMENT: Calculate real-world scale (Width, Height, Depth in cm). Look for anchors like hands (~18cm) or floor tiles (~30cm).
2. CONDITION: Assign [Brand New, Excellent, Good, Fair, Poor] based on visual inspection.
3. VALUATION: Estimate a Resale Price in USD based on brand and condition.
4. 3D TEMPLATE: Map to closest template: ["tv", "chair", "laptop", "lamp", "sofa", "table", "sneaker", "generic_box"].
5. MARKETING: 1-sentence headline and 3 bullet points of visual features.
"""

SYSTEM_PROMPT_3D_EXTRACT = """
You are a 3D modeling AI. Your objective is to analyze a video of an object and generate a low-poly 3D model representation of it.

### DIRECTIVES:
1. Identify the primary object in the video.
2. Describe its shape and structure.
3. Generate a valid .obj file string (using 'v' for vertices and 'f' for faces) that represents a low-poly approximation (e.g., a bounding box or basic geometric primitive) of the object's shape and proportions.
4. Provide a highly detailed texture prompt that could be used by a text-to-image or text-to-3D texture generator to paint this model.
"""

@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "AuraMarket Backend is running!"}

@app.post("/analyze")
async def analyze_video(product_hint: str = Form(...), video: UploadFile = File(...)) -> Any:
    """
    Analyzes an uploaded video using Gemini 1.5 Flash to extract spatial dimensions,
    market valuation, and 3D rendering data (Template Scaling Approach).
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{video.filename}")
    temp_path = temp_file.name
    
    try:
        content = await video.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        print(f"Uploading {video.filename} to Gemini...")
        uploaded_file = client.files.upload(file=temp_path)

        print("Waiting for video processing...")
        while uploaded_file.state == "PROCESSING":
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)
            
        if uploaded_file.state == "FAILED":
            raise Exception("Gemini video processing failed.")

        print("Generating spatial data...")
        user_prompt = f"Hint from seller: {product_hint}. Analyze video."
        
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[uploaded_file, user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_ANALYZE,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=DigitalTwin,
            )
        )
        
        client.files.delete(name=uploaded_file.name)
        return json.loads(response.text)

    except Exception as e:
        if "uploaded_file" in locals() and uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/extract-3d")
async def extract_3d_model(video: UploadFile = File(...)) -> Any:
    """
    Hybrid Approach: Extracts object details from video using Gemini
    and generates a low-poly .obj 3D model representation directly.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{video.filename}")
    temp_path = temp_file.name
    
    try:
        content = await video.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        print(f"Uploading {video.filename} to Gemini for 3D extraction...")
        uploaded_file = client.files.upload(file=temp_path)

        print("Waiting for video processing...")
        while uploaded_file.state == "PROCESSING":
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)
            
        if uploaded_file.state == "FAILED":
            raise Exception("Gemini video processing failed.")

        print("Generating 3D model data...")
        user_prompt = "Analyze this video, extract the main object, and generate its 3D .obj representation."
        
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[uploaded_file, user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_3D_EXTRACT,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=Extracted3DModel,
            )
        )
        
        client.files.delete(name=uploaded_file.name)
        return json.loads(response.text)

    except Exception as e:
        if "uploaded_file" in locals() and uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
