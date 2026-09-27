# 🚀 AuraMarket Backend (Spatially AI)

AuraMarket turns a short smartphone video of an item into a ready-to-publish marketplace listing. Gemini identifies the item, researches it with Google Search, creates a studio product photo, and builds a real-world-size 3D model that can be placed in a room with AR.

> **✨ Note:** This backend architecture and codebase was built with the help of **Gemini 3.1 Pro**.

## 🏗️ Technical Architecture

- **Framework:** Python 3.12 + FastAPI
- **AI Engine:** Google Gemini (using the `google-genai` SDK)
- **Deployment:** Google Cloud Run
- **Scan pipeline (`POST /scan`):**
  1. **Frames:** the browser samples about 20 JPEG frames from the video and uploads them. A raw video is also accepted and sampled on the server with a bundled ffmpeg (`imageio-ffmpeg`).
  2. **Identify** (`gemini-3.8-flash`): item name, brand and model, category, condition, estimated size and price, and the best frame with a bounding box.
  3. Then, in parallel:
     - **Research** (`gemini-3.8-flash` with Google Search grounding, through the Interactions API): official dimensions and specs, original retail price, used-market price range, description and highlights, with sources.
     - **Studio photo** (`gemini-3.1-flash-image`, "Nano Banana 2"): a clean product photo made from the best frame. Falls back to the cropped frame if image generation fails.
     - **3D model** (`gemini-3.8-flash`): a structured plan of 15 to 45 basic shapes (boxes, rounded boxes, cylinders, cones, spheres, capsules, tori and lathe profiles) with real proportions, colors and materials.
  4. **Finalize:** manufacturer dimensions replace the video estimate when they plausibly match.
- **3D models:** Gemini cannot output mesh files, so it outputs a shape plan that `model_builder.py` turns into a GLB with `trimesh`. The plan is packed into a compact token, and `GET /model.glb` rebuilds the GLB from that token at the requested size. The service stays stateless (no storage bucket), and editing a listing's dimensions simply changes the model URL.

| File | Purpose |
| --- | --- |
| `main.py` | FastAPI app: upload limits, NDJSON streaming, GLB endpoint |
| `pipeline.py` | Gemini calls (identify, research, photo, geometry) and result assembly |
| `schemas.py` | Pydantic models used as Gemini JSON schemas |
| `model_builder.py` | Shape-plan validation, recipe tokens and GLB export |
| `media.py` | Frame normalization, video sampling, cropping |

## 🚀 Getting Started (Local Development)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Dichill/AuraMarket-backend.git
   cd AuraMarket-backend
   ```

2. **Set up a virtual environment (Python 3.12):**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

   Optional model overrides (defaults shown):

   | Variable | Default |
   | --- | --- |
   | `GEMINI_VISION_MODEL` | `gemini-3.8-flash` |
   | `GEMINI_RESEARCH_MODEL` | `gemini-3.8-flash` |
   | `GEMINI_GEOMETRY_MODEL` | `gemini-3.8-flash` |
   | `GEMINI_GEOMETRY_THINKING` | `medium` (`low` is faster with fewer parts, `high` is slower with more detail) |
   | `GEMINI_IMAGE_MODEL` | `gemini-3.1-flash-image` |
   | `GEMINI_FALLBACK_MODEL` | `gemini-3.1-pro-preview` (used when a text step fails) |
   | `GEMINI_FALLBACK_IMAGE_MODEL` | `gemini-3-pro-image` (used when the photo step fails) |

5. **Run the server:**
   ```bash
   uvicorn main:app --reload --port 8080
   ```
   The API will be available at `http://localhost:8080`. Point the frontend at it with `VITE_BACKEND_URL=http://localhost:8080`.

## 📚 API Documentation

### 1. Health Check
- **Endpoints:** `GET /` and `GET /health`
- **Response:** service status, version and the configured Gemini models.

### 2. Scan a Video
- **Endpoint:** `POST /scan`
- **Content-Type:** `multipart/form-data`
- **Payload:**
  - `frames` (files, repeatable): up to 24 images sampled from the video, 6 MB each. **Or:**
  - `video` (file): one video up to 30 MB (MP4, MOV, M4V, WebM, MKV, AVI, 3GP). Cloud Run caps request bodies at 32 MB, so sending frames is preferred.
  - `product_hint` (string, optional): e.g. "Sony Bravia 55-inch TV". Up to 200 characters.
- **Response:** `application/x-ndjson`, one JSON event per line:
  - `{"type": "progress", "stage": "identify", "status": "running", "message": "..."}`. Stages: `frames`, `identify`, `research`, `photo`, `model`, `finalize`. Statuses: `running`, `done`, `failed`.
  - `{"type": "ping"}` whenever the pipeline has been quiet for 10 seconds.
  - `{"type": "error", "stage": "...", "message": "..."}` when the scan cannot finish (for example, no object found, or no 3D model could be built).
  - `{"type": "result", ...}` with:
    - `listing`: title, verified name, brand, model number, category, condition and notes, description, features, specs, suggested price with used range and MSRP, dimensions in cm and whether they came from the manufacturer or the video.
    - `model`: `recipeToken`, `partCount`, `naturalSize` and a ready-made `modelPath`.
    - `thumbnail`: JPEG data URL and whether it is a `studio` photo or a `video_frame`.
    - `research`: sources, the Google searches that were run, and Google's search-suggestions HTML, which must be shown alongside grounded results.
    - `warnings`, `models` (which model handled each step) and `timingsMs`.

### 3. Download a Generated Model
- **Endpoint:** `GET /model.glb?v=1&w=<cm>&h=<cm>&d=<cm>&r=<recipeToken>` (also `HEAD`)
- `w`, `h` and `d` are optional, but must be given together. The model is scaled to exactly that width, height and depth, centered, resting on the floor, and exported in meters for AR.
- **Response:** `model/gltf-binary`, gzip-compressed when the client accepts it, cached as immutable. Invalid tokens return `400`.

## ☁️ Deployment (Google Cloud Run)

Pushing to `master` redeploys the service through Cloud Run continuous deployment. To deploy manually with the Google Cloud CLI:

```bash
gcloud run deploy auramarket-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY="your_api_key_here"
```
