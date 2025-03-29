from pathlib import Path
from typing import Optional
import io
import math

import pillow_heif
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, Response
from huggingface_hub import hf_hub_download
from PIL import Image
from refiners.fluxion.utils import manual_seed
from refiners.foundationals.latent_diffusion import Solver, solvers
from pydantic import BaseModel
import gc

from enhancer import ESRGANUpscaler, ESRGANUpscalerCheckpoints

# Register HEIF and AVIF support
pillow_heif.register_heif_opener()
pillow_heif.register_avif_opener()

# Initialize FastAPI app
app = FastAPI(
    title="Fashable.AI Finegrain Image Enhancer API",
    description="API for enhancing low resolution images into high resolution versions with added generated details",
    version="1.0.0"
)

# Define request model
class EnhancementRequest(BaseModel):
    prompt: str = "masterpiece, best quality, highres"
    negative_prompt: str = "worst quality, low quality, normal quality"
    seed: int = 42
    upscale_factor: float = 2.0
    controlnet_scale: float = 0.6
    controlnet_decay: float = 1.0
    condition_scale: int = 6
    tile_width: int = 112
    tile_height: int = 144
    denoise_strength: float = 0.35
    num_inference_steps: int = 18
    solver: str = "DDIM"

# Load checkpoints
CHECKPOINTS = ESRGANUpscalerCheckpoints(
    unet=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.unet",
            filename="model.safetensors",
            revision="347d14c3c782c4959cc4d1bb1e336d19f7dda4d2",
        )
    ),
    clip_text_encoder=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.text_encoder",
            filename="model.safetensors",
            revision="744ad6a5c0437ec02ad826df9f6ede102bb27481",
        )
    ),
    lda=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.autoencoder",
            filename="model.safetensors",
            revision="3c1aae3fc3e03e4a2b7e0fa42b62ebb64f1a4c19",
        )
    ),
    controlnet_tile=Path(
        hf_hub_download(
            repo_id="refiners/controlnet.sd1_5.tile",
            filename="model.safetensors",
            revision="48ced6ff8bfa873a8976fa467c3629a240643387",
        )
    ),
    esrgan=Path(
        hf_hub_download(
            repo_id="philz1337x/upscaler",
            filename="4x-UltraSharp.pth",
            revision="011deacac8270114eb7d2eeff4fe6fa9a837be70",
        )
    ),
    negative_embedding=Path(
        hf_hub_download(
            repo_id="philz1337x/embeddings",
            filename="JuggernautNegative-neg.pt",
            revision="203caa7e9cc2bc225031a4021f6ab1ded283454a",
        )
    ),
    negative_embedding_key="string_to_param.*",
    loras={
        "more_details": Path(
            hf_hub_download(
                repo_id="philz1337x/loras",
                filename="more_details.safetensors",
                revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
            )
        ),
        "sdxl_render": Path(
            hf_hub_download(
                repo_id="philz1337x/loras",
                filename="SDXLrender_v2.0.safetensors",
                revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
            )
        ),
    },
)

# Initialize the enhancer
DEVICE_CPU = torch.device("cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
enhancer = ESRGANUpscaler(checkpoints=CHECKPOINTS, device=DEVICE_CPU, dtype=DTYPE)

# Move enhancer to GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enhancer.to(device=DEVICE, dtype=DTYPE)

# Base memory required for model loading (in GB)
BASE_MEMORY = 4

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    return 0

def estimate_memory_usage(image_size, upscale_factor):
    # Rough estimation of memory usage in GB
    # This is a conservative estimate based on typical model memory requirements
    image_memory = (image_size[0] * image_size[1] * 3 * 4 * upscale_factor**2) / (1024**3)  # RGB image in float32
    return BASE_MEMORY + image_memory

def calculate_optimal_tile_size(image_size):
    # Calculate tile size that's divisible by 8 (model requirement)
    # and appropriate for the image size
    width, height = image_size
    tile_width = min(112, ((width + 7) // 8) * 8)
    tile_height = min(144, ((height + 7) // 8) * 8)
    return tile_width, tile_height

@app.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    prompt: str = Form("masterpiece, best quality, highres"),
    negative_prompt: str = Form("worst quality, low quality, normal quality"),
    seed: int = Form(1337),
    upscale_factor: float = Form(2.0),
    controlnet_scale: float = Form(0.6),
    controlnet_decay: float = Form(1.0),
    condition_scale: int = Form(6),
    tile_width: int = Form(112),
    tile_height: int = Form(144),
    denoise_strength: float = Form(0.35),
    num_inference_steps: int = Form(18),
    solver: str = Form("DDIM", description="The solver to use for the image enhancement. Default is DDIM. (DPMSolver, DDPM, DDIM, Euler, FrankenSolver, LCMSolver, ModelPredictionType, NoiseSchedule, TimestepSpacing )") 
):
    
    try:
        # Read and validate image
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents))
        
        # Calculate optimal tile size based on input image
        optimal_tile_width, optimal_tile_height = calculate_optimal_tile_size(input_image.size)
        tile_width = min(tile_width, optimal_tile_width)
        tile_height = min(tile_height, optimal_tile_height)
        
        # Check image size and upscale factor
        image_size = input_image.size
        estimated_memory = estimate_memory_usage(image_size, upscale_factor)
        available_memory = get_gpu_memory()
        
        # More conservative memory threshold for larger upscale factors
        memory_threshold = 0.7 if upscale_factor > 2.0 else 0.8
        
        if estimated_memory > available_memory * memory_threshold:
            # Calculate maximum safe upscale factor
            max_upscale = math.sqrt((available_memory * memory_threshold - BASE_MEMORY) * (1024**3) / (image_size[0] * image_size[1] * 3 * 4))
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Requested upscale factor {upscale_factor} would exceed available memory. "
                    f"For this image size ({image_size[0]}x{image_size[1]}), "
                    f"maximum recommended upscale factor is {min(2.0, max_upscale):.1f}. "
                    "Try reducing the upscale factor or image size."
                )
            )

        # Set random seed
        manual_seed(seed)
        
        # Get solver type
        solver_type: type[Solver] = getattr(solvers, solver)
        
        # Clear CUDA cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Process image with adjusted tile size
        enhanced_image = enhancer.upscale(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            upscale_factor=upscale_factor,
            controlnet_scale=controlnet_scale,
            controlnet_scale_decay=controlnet_decay,
            condition_scale=condition_scale,
            tile_size=(tile_height, tile_width),  # Using adjusted tile size
            denoise_strength=denoise_strength,
            num_inference_steps=num_inference_steps,
            loras_scale={"more_details": 0.5, "sdxl_render": 1.0},
            solver_type=solver_type,
        )
        
        # Clear CUDA cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Convert enhanced image to bytes
        img_byte_arr = io.BytesIO()
        enhanced_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return Response(
            content=img_byte_arr,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=enhanced_{file.filename}"
            }
        )
        
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        raise HTTPException(
            status_code=400,
            detail=(
                "Out of memory error. For this image size, try:\n"
                "1. Reducing the upscale factor to 2.0 or less\n"
                "2. Reducing the input image size\n"
                "3. Using smaller tile sizes (try 64x64)"
            )
        )
    except RuntimeError as exp:
        if "size of tensor" in str(exp):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Image size incompatible with tile size. Try:\n"
                    "1. Using smaller tile sizes (multiple of 8)\n"
                    "2. Resizing the input image to a larger size"
                )
            )
        raise HTTPException(status_code=500, detail=str(exp))

@app.get("/")
async def root():
    return {
        "message": "Welcome to Fashable Finegrain Image Enhancer API",
        "endpoints": {
            "/enhance": "POST endpoint for image enhancement"
        }
    } 