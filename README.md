# Fashable Finegrain Image Enhancer

A powerful image enhancement tool that turns low-resolution images into high-resolution versions with added generated details. This project provides both a Gradio web interface and a FastAPI REST API.

## Features

- Upscale images with AI-generated details
- Support for various image formats (PNG, JPEG, HEIF, AVIF)
- Customizable enhancement parameters
- Both web interface and REST API available
- GPU acceleration support
- High-quality upscaling using ESRGAN and ControlNet

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/imageenhancer.git
cd imageenhancer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
Or Better use UV 
```bash
uv sync
```

## Usage

### Web Interface (Gradio)

To run the web interface:

```bash
python src/app.py
```

This will start a local web server with the Gradio interface. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:7860).

### REST API (FastAPI)

To run the API server:

```bash
uvicorn src.api:app --reload
```

The API will be available at http://localhost:8000

#### API Endpoints

1. `GET /`: Welcome message and available endpoints
2. `POST /enhance`: Image enhancement endpoint

#### Example API Usage

Using curl:
```bash
curl -X POST "http://localhost:8000/enhance" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg" \
  -F "request={\"prompt\":\"masterpiece, best quality, highres\",\"upscale_factor\":2.0}" \
  --output enhanced_image.png
```

Using Python requests:
```python
import requests

url = "http://localhost:8000/enhance"
files = {"file": open("your_image.jpg", "rb")}
data = {
    "request": {
        "prompt": "masterpiece, best quality, highres",
        "upscale_factor": 2.0
    }
}

response = requests.post(url, files=files, json=data)
with open("enhanced_image.png", "wb") as f:
    f.write(response.content)
```

### Enhancement Parameters

The following parameters can be customized:

- `prompt`: Text prompt for image enhancement (default: "masterpiece, best quality, highres")
- `negative_prompt`: Text prompt for what to avoid (default: "worst quality, low quality, normal quality")
- `seed`: Random seed for reproducibility (default: 42)
- `upscale_factor`: Image upscaling factor (default: 2.0)
- `controlnet_scale`: ControlNet influence strength (default: 0.6)
- `controlnet_decay`: ControlNet scale decay (default: 1.0)
- `condition_scale`: Conditioning scale (default: 6)
- `tile_width`: Width of processing tiles (default: 112)
- `tile_height`: Height of processing tiles (default: 144)
- `denoise_strength`: Denoising strength (default: 0.35)
- `num_inference_steps`: Number of inference steps (default: 18)
- `solver`: Solver type (default: "DDIM")

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- Sufficient RAM (8GB minimum recommended)


## Acknowledgments

- Built with [Refiners](https://github.com/finegrain-ai/refiners)
- Uses ESRGAN for upscaling
- Powered by ControlNet for detail enhancement
