# 🖼️ AI Art and Real Image Detector

A FastAPI-based REST API that classifies an uploaded image as **Real**, **AI Generated**, or **Other / Uncertain**, powered by a fine-tuned TensorFlow/Keras (Xception-based) image classification model.

## Features

- Single REST endpoint (`/predict`) for image classification
- TensorFlow/Keras model served on CPU, no GPU required
- Built-in confidence thresholding to flag uncertain predictions as `"Other Image"`
- Concurrency-limited inference (max 2 simultaneous requests) to keep memory usage predictable
- Request timeout protection (25s) to avoid hanging requests
- CORS enabled for use from any frontend
- Docker-ready, pre-configured for Hugging Face Spaces (port 7860)

## Tech Stack

| Component       | Technology                          |
|------------------|--------------------------------------|
| API Framework    | FastAPI + Uvicorn                   |
| ML Framework     | TensorFlow / Keras                  |
| Image Processing | OpenCV, Pillow, NumPy               |
| Model Format     | `.keras` (Xception-based, 224x224 input) |
| Deployment       | Docker (Hugging Face Spaces compatible) |

## Project Structure

```
Ai_art_and_real_image_detector/
├── main.py             # FastAPI application and inference logic
├── best_model.keras     # Trained classification model
├── requirements.txt      # Python dependencies
├── Dockerfile          # Container build configuration
└── README.md            # Project documentation
```

## Prerequisites

- Python 3.10 or higher
- pip
- (Optional) Docker, if you prefer a containerized setup
- ~1 GB free disk space (model + TensorFlow dependencies)

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Ai_art_and_real_image_detector
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the model file

Ensure `best_model.keras` is present in the project root. The application will still start without it, but the API will return a `model_error` status until a valid model is added.

### 5. Run the API server locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

## Running with Docker

### 1. Build the image

```bash
docker build -t ai-image-detector .
```

### 2. Run the container

```bash
docker run -p 7860:7860 ai-image-detector
```

The Dockerfile is pre-configured to expose port `7860`, matching Hugging Face Spaces' requirements. If running locally and you prefer a different port, adjust the `EXPOSE` and `CMD` lines in the `Dockerfile`, or map ports with `-p <host_port>:7860`.

## API Reference

### `GET /`

Health check endpoint. Returns `200` with `{"status": "ok"}` if the model loaded successfully, or `500` with the load error otherwise.

### `GET /health`

Identical behavior to `/`, provided as a conventional health-check path for deployment platforms.

### `POST /predict`

Classifies an uploaded image.

**Request:** `multipart/form-data` with a single field, `file`, containing the image.

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg"
```

**Response:**

```json
{
  "filename": "image.jpg",
  "prediction": "AI Generated",
  "confidence": "92.31%"
}
```

`prediction` will be one of:
- `Real`
- `AI Generated`
- `Other Image` (returned when the model's confidence is below the 60% threshold for either class)

**Error responses:**

| Status | Cause                                  |
|--------|------------------------------------------|
| `400`  | Uploaded file is not a valid image        |
| `500`  | Model failed to load, or inference error occurred |
| `504`  | Inference exceeded the 25-second timeout   |

## Configuration Notes

- **Inference concurrency** is capped at 2 simultaneous requests via an `asyncio.Semaphore`. Adjust the `Semaphore(2)` value in `main.py` if deploying on a larger instance.
- **Confidence threshold** for a definitive `Real`/`AI Generated` label is `0.60` (`TH` in `main.py`). Lower it to make the model more decisive, or raise it to make `Other Image` results more common.
- **Input size** for the model is fixed at `224x224`; images are automatically resized.

## Deploying to Hugging Face Spaces

1. Create a new Space and select **Docker** as the SDK.
2. Push this repository's contents (including `Dockerfile`, `main.py`, `requirements.txt`, and `best_model.keras`) to the Space's git repository.
3. The Space will build the Docker image automatically and expose the API on port `7860`.

## Troubleshooting

| Issue                                   | Likely Cause / Fix                                              |
|-------------------------------------------|---------------------------------------------------------------------|
| `model_error` shown at `/` or `/health`   | `best_model.keras` is missing, corrupted, or incompatible with the installed TensorFlow version |
| `500: Inference failed`                  | Check the uploaded file is a valid, decodable image                |
| Slow first request                        | The model performs a warmup prediction on startup; subsequent requests are faster |
| `opencv-python-headless` install fails     | Ensure you're on Python 3.10+ and have an up-to-date `pip`        |

## License

Add your preferred license here (e.g., MIT, Apache 2.0).
