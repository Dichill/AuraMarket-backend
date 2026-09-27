# 🚀 AuraMarket Backend (Spatially AI)

AuraMarket is a multimodal "Spatial E-commerce Engine" that democratizes Augmented Reality (AR) for everyday sellers. By leveraging Gemini 1.5 Flash’s ability to process video, we turn a simple 10-second smartphone video into a fully scaled, shoppable 3D AR digital twin—instantly and for zero cost.

> **✨ Note:** This backend architecture and codebase was built with the help of **Gemini 3.1 Pro**.

## 🏗️ Technical Architecture

- **Framework:** Python + FastAPI
- **AI Engine:** Google Gemini (using the new `google-genai` SDK)
- **Deployment:** Google Cloud Run
- **Core Functionality:** 
  - **Template Scaling (`/analyze`):** Extracts real-world dimensions and maps them to generic 3D templates for instant, zero-latency AR rendering.
  - **Hybrid 3D Extraction (`/extract-3d`):** Directly generates low-poly 3D mesh (`.obj`) data and texture prompts from video input.

## 🚀 Getting Started (Local Development)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Dichill/AuraMarket-backend.git
   cd AuraMarket-backend
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
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

5. **Run the server:**
   ```bash
   uvicorn main:app --reload --port 8080
   ```
   The API will be available at `http://localhost:8080`.

## 📚 API Documentation

### 1. Health Check
- **Endpoint:** `GET /`
- **Response:** `{"status": "AuraMarket Backend is running!"}`

### 2. Analyze Video (Template Scaling)
- **Endpoint:** `POST /analyze`
- **Content-Type:** `multipart/form-data`
- **Payload:** 
  - `product_hint` (string): e.g., "Sony Bravia TV"
  - `video` (file): The video file to analyze.
- **Response:** Returns a strict JSON "Digital Twin" containing spatial data, market analysis, marketing copy, and a 3D template scale vector.

### 3. Extract 3D Model (Hybrid Approach)
- **Endpoint:** `POST /extract-3d`
- **Content-Type:** `multipart/form-data`
- **Payload:** 
  - `video` (file): The video file to analyze.
- **Response:** Returns a JSON object containing the object name, description, raw `.obj` file content, and a recommended texture prompt.

## ☁️ Deployment (Google Cloud Run)

To deploy to Google Cloud Run, use the Google Cloud CLI:

```bash
gcloud run deploy auramarket-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY="your_api_key_here"
```
