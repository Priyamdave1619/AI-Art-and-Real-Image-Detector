---
title: AI Art and Real Image Detector
emoji: 🖼️
colorFrom: red
colorTo: green
sdk: docker
pinned: false
---

## 🚀 AI Art vs Real Image Detector (FastAPI)

This project detects whether an uploaded image is:
- **Real**
- **AI Generated**
- **Other / Uncertain**

Built using:
- FastAPI
- TensorFlow (CPU)
- OpenCV
- Docker (Hugging Face Spaces compatible)

---

## 🧠 Model
- Format: `.keras`
- Input size: `224 x 224`
- Output: single probability score
---
## 🔌 API Endpoints

### `GET /`
Health check

### `GET /health`
Health check with model status

### `POST /predict`
Upload an image file

**Response**
```json
{
  "filename": "image.jpg",
  "prediction": "AI Generated",
  "confidence": "92.31%"
}