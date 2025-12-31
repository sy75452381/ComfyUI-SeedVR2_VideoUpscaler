#!/usr/bin/env python3
"""
SeedVR2 Image Upscaler - Web UI and API

A FastAPI-based web interface and REST API for high-quality image upscaling
using SeedVR2 diffusion models. Focused on single image processing.

Features:
    • RESTful API with file upload support
    • Modern web interface with drag-and-drop
    • Real-time progress updates via WebSocket
    • Configurable upscaling parameters
    • GPU memory optimization options

Usage:
    python webui.py --port 8000
    python webui.py --host 0.0.0.0 --port 7860  # For public access
"""

import os
import sys
import io
import base64
import time
import uuid
import asyncio
import argparse
import platform
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Set up path before any other imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Configure platform-specific memory management before heavy imports
if platform.system() == "Darwin":
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

# FastAPI and web dependencies
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Heavy imports after environment configuration
import torch
import cv2
import numpy as np
import imageio
import imagecodecs

# Project imports
from src.utils.downloads import download_weight
from src.utils.model_registry import get_available_dit_models, DEFAULT_DIT, DEFAULT_VAE, MODEL_REGISTRY
from src.utils.constants import SEEDVR2_FOLDER_NAME
from src.core.generation_utils import (
    setup_generation_context,
    prepare_runner,
    compute_generation_info,
    log_generation_start,
    load_text_embeddings,
    script_directory
)
from src.core.generation_phases import (
    encode_all_batches,
    upscale_all_batches,
    decode_all_batches,
    postprocess_all_batches
)
from src.utils.debug import Debug
from src.optimization.memory_manager import clear_memory, get_gpu_backend

# Initialize debug instance
debug = Debug(enabled=False)

# WebUI-specific default model (keeps CLI defaults unchanged)
DEFAULT_DIT_WEBUI = "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"

# Global state for model caching
model_cache: Dict[str, Any] = {}
current_loaded_model: Optional[str] = None  # Track which DiT model is loaded

# Ensure outputs directory exists
OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI(
    title="SeedVR2 Image Upscaler",
    description="High-quality AI image upscaling using SeedVR2 diffusion models",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


# =============================================================================
# Core Processing Functions
# =============================================================================

def extract_image_tensor(image_bytes: bytes) -> torch.Tensor:
    """
    Convert image bytes to tensor format for processing.

    Args:
        image_bytes: Raw image bytes from upload

    Returns:
        Image tensor [1, H, W, C] in Float16, range [0,1]
    """
    # Decode image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if frame is None:
        raise ValueError("Could not decode image file")

    # Convert BGR(A) to RGB(A)
    if len(frame.shape) == 3:
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        # Grayscale - convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # Normalize to [0, 1] and convert to tensor
    frame = frame.astype(np.float32) / 255.0
    frames_tensor = torch.from_numpy(frame[None, ...]).to(torch.float16)

    return frames_tensor


def tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
    """
    Convert output tensor to PNG bytes.

    Args:
        tensor: Image tensor [T, H, W, C] in range [0,1]

    Returns:
        PNG-encoded bytes
    """
    # Get first frame (for single image processing)
    frame = tensor[0].cpu().numpy()

    # Convert to uint8
    frame_uint8 = (frame * 255.0).clip(0, 255).astype(np.uint8)

    # Handle RGBA vs RGB
    if frame_uint8.shape[-1] == 4:
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGBA2BGRA)
    else:
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)

    # Encode as PNG
    success, encoded = cv2.imencode('.png', frame_bgr)
    if not success:
        raise ValueError("Failed to encode output image")

    return encoded.tobytes()


def save_image_to_disk(tensor: torch.Tensor, format: str) -> str:
    """
    Save image tensor to disk in specified format.

    Args:
        tensor: Image tensor [1, H, W, C]
        format: Output format ('png', 'jpeg', 'jxl')

    Returns:
        Filename of saved image
    """
    # Get frame
    frame = tensor[0].cpu().numpy()
    frame_uint8 = (frame * 255.0).clip(0, 255).astype(np.uint8)

    # Generate filename
    ext = format.lower()
    if ext == 'jpeg': ext = 'jpg'
    filename = f"upscaled_{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(OUTPUTS_DIR, filename)

    if format == 'jxl':
        # Use imageio with imagecodecs for JXL
        try:
            # ImageIO expects RGB, not BGR
            # Our tensor is RGB, but if we used cv2 conversions before we need to be careful.
            # tensor is [H, W, C] RGB(A)
            imageio.imwrite(filepath, frame_uint8, format='jxl')
        except Exception as e:
            # Fallback if possible or raise
            raise ValueError(f"Failed to save JXL: {e}")
    else:
        # Use OpenCV for standard formats
        if frame_uint8.shape[-1] == 4:
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGBA2BGRA)
        else:
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)

        if format == 'jpeg' or format == 'jpg':
            cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(filepath, frame_bgr)

    return filename


def _device_id_to_name(device_id: str, platform_type: str = None) -> str:
    """Convert device ID to full device name."""
    if device_id in ("cpu", "none"):
        return device_id

    if platform_type is None:
        platform_type = get_gpu_backend()

    if platform_type == "mps":
        return "mps"

    return f"{platform_type}:{device_id}"


def _parse_offload_device(offload_arg: str, platform_type: str = None, cache_enabled: bool = False) -> Optional[str]:
    """Parse offload device argument to full device name."""
    if offload_arg == "none":
        return "cpu" if cache_enabled else None

    if offload_arg == "cpu":
        return "cpu"

    if ":" in offload_arg:
        return offload_arg

    return _device_id_to_name(offload_arg, platform_type)


def process_image_upscale(
    image_tensor: torch.Tensor,
    resolution: int = 2160,
    max_resolution: int = 4096,
    dit_model: str = DEFAULT_DIT_WEBUI,
    model_dir: Optional[str] = None,
    blocks_to_swap: int = 0,
    swap_io_components: bool = False,
    vae_encode_tiled: bool = True,
    vae_decode_tiled: bool = True,
    vae_tile_size: int = 1024,
    vae_tile_overlap: int = 128,
    color_correction: str = "lab",
    seed: int = 42,
    cache_models: bool = True,
    progress_callback: Optional[callable] = None
) -> torch.Tensor:
    """
    Process a single image through the SeedVR2 upscaling pipeline.

    Args:
        image_tensor: Input image tensor [1, H, W, C]
        resolution: Target resolution for shortest edge
        max_resolution: Maximum resolution for any edge (0 = no limit)
        dit_model: DiT model to use
        model_dir: Model directory path
        blocks_to_swap: Number of transformer blocks to swap for VRAM savings
        swap_io_components: Whether to offload DiT I/O layers
        vae_encode_tiled: Enable tiled VAE encoding
        vae_decode_tiled: Enable tiled VAE decoding
        vae_tile_size: Tile size for VAE tiling
        vae_tile_overlap: Tile overlap for VAE tiling
        color_correction: Color correction method
        seed: Random seed
        cache_models: Whether to cache models between requests
        progress_callback: Optional callback for progress updates

    Returns:
        Upscaled image tensor [1, H', W', C]
    """
    global model_cache, current_loaded_model

    # Check if model has changed - if so, clear the cache first
    if current_loaded_model is not None and current_loaded_model != dit_model:
        debug.log(f"Model changed from {current_loaded_model} to {dit_model}, clearing cache...",
                  category="cache", force=True)
        model_cache = {}
        clear_memory(debug=debug, deep=True, force=True)
        current_loaded_model = None

    # Determine platform and device
    platform_type = get_gpu_backend()
    device_id = "0"
    inference_device = _device_id_to_name(device_id, platform_type)

    # Configure offload devices
    # When caching models, we keep them on GPU (no offload) unless BlockSwap is used
    # This ensures fast subsequent requests by avoiding CPU<->GPU transfers
    if cache_models and blocks_to_swap == 0:
        # Keep models on GPU for fast repeated inference
        dit_offload = None
        vae_offload = None
    else:
        # Offload to CPU when using BlockSwap or not caching
        dit_offload = _parse_offload_device("cpu" if blocks_to_swap > 0 else "none", platform_type, cache_models)
        vae_offload = _parse_offload_device("cpu" if cache_models else "none", platform_type, cache_models)
    tensor_offload = _parse_offload_device("cpu", platform_type, False)

    # Model directory
    if model_dir is None:
        model_dir = f"./models/{SEEDVR2_FOLDER_NAME}"

    # Setup or reuse generation context
    if cache_models and 'ctx' in model_cache:
        ctx = model_cache['ctx']
        # Clear previous run data but keep device config
        keys_to_keep = {'dit_device', 'vae_device', 'dit_offload_device',
                       'vae_offload_device', 'tensor_offload_device', 'compute_dtype'}
        for key in list(ctx.keys()):
            if key not in keys_to_keep:
                del ctx[key]
    else:
        ctx = setup_generation_context(
            dit_device=inference_device,
            vae_device=inference_device,
            dit_offload_device=dit_offload,
            vae_offload_device=vae_offload,
            tensor_offload_device=tensor_offload,
            debug=debug
        )
        if cache_models:
            model_cache['ctx'] = ctx

    # Prepare runner with caching support
    dit_id = "webui_dit" if cache_models else None
    vae_id = "webui_vae" if cache_models else None

    runner, cache_context = prepare_runner(
        dit_model=dit_model,
        vae_model=DEFAULT_VAE,
        model_dir=model_dir,
        debug=debug,
        ctx=ctx,
        dit_cache=cache_models,
        vae_cache=cache_models,
        dit_id=dit_id,
        vae_id=vae_id,
        block_swap_config={
            'blocks_to_swap': blocks_to_swap,
            'swap_io_components': swap_io_components,
            'offload_device': dit_offload,
        } if blocks_to_swap > 0 else None,
        encode_tiled=vae_encode_tiled,
        encode_tile_size=(vae_tile_size, vae_tile_size),
        encode_tile_overlap=(vae_tile_overlap, vae_tile_overlap),
        decode_tiled=vae_decode_tiled,
        decode_tile_size=(vae_tile_size, vae_tile_size),
        decode_tile_overlap=(vae_tile_overlap, vae_tile_overlap),
        tile_debug="false",
        attention_mode="sdpa"
    )

    ctx['cache_context'] = cache_context
    if cache_models:
        model_cache['runner'] = runner
        current_loaded_model = dit_model  # Track loaded model for cache invalidation

    # Load text embeddings
    ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['dit_device'], ctx['compute_dtype'], debug)

    # Compute generation info
    image_tensor, gen_info = compute_generation_info(
        ctx=ctx,
        images=image_tensor,
        resolution=resolution,
        max_resolution=max_resolution,
        batch_size=1,
        uniform_batch_size=False,
        seed=seed,
        prepend_frames=0,
        temporal_overlap=0,
        debug=debug
    )
    log_generation_start(gen_info, debug)

    # Progress updates
    def progress_wrapper(current, total, frames, phase):
        if progress_callback:
            progress_callback(phase, current, total)

    # Phase 1: Encode
    if progress_callback:
        progress_callback("Encoding", 0, 1)
    ctx = encode_all_batches(
        runner, ctx=ctx, images=image_tensor,
        debug=debug,
        batch_size=1,
        uniform_batch_size=False,
        seed=seed,
        progress_callback=progress_wrapper,
        temporal_overlap=0,
        resolution=resolution,
        max_resolution=max_resolution,
        input_noise_scale=0.0,
        color_correction=color_correction
    )

    # Phase 2: Upscale
    if progress_callback:
        progress_callback("Upscaling", 0, 1)
    ctx = upscale_all_batches(
        runner, ctx=ctx, debug=debug,
        progress_callback=progress_wrapper,
        seed=seed,
        latent_noise_scale=0.0,
        cache_model=cache_models
    )

    # Phase 3: Decode
    if progress_callback:
        progress_callback("Decoding", 0, 1)
    ctx = decode_all_batches(
        runner, ctx=ctx, debug=debug,
        progress_callback=progress_wrapper,
        cache_model=cache_models
    )

    # Phase 4: Post-process
    if progress_callback:
        progress_callback("Post-processing", 0, 1)
    ctx = postprocess_all_batches(
        ctx=ctx, debug=debug,
        progress_callback=progress_wrapper,
        color_correction=color_correction,
        prepend_frames=0,
        temporal_overlap=0,
        batch_size=1
    )

    result_tensor = ctx['final_video']

    # Convert to CPU and compatible dtype
    if result_tensor.is_cuda or (hasattr(result_tensor, 'is_mps') and result_tensor.is_mps):
        result_tensor = result_tensor.cpu()
    if result_tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        result_tensor = result_tensor.to(torch.float32)

    return result_tensor


# =============================================================================
# API Endpoints
# =============================================================================

def _describe_dit_model(model_name: str) -> Dict[str, Any]:
    """Build a small metadata payload for UI display."""
    lower = model_name.lower()
    model_info = MODEL_REGISTRY.get(model_name)

    # Best-effort size detection (covers both registry + discovered-on-disk models)
    size = None
    if model_info and getattr(model_info, "size", None):
        size = model_info.size
    elif "7b" in lower:
        size = "7B"
    elif "3b" in lower:
        size = "3B"

    precision = None
    variant = None
    repo = None
    description = None

    if model_info:
        precision = model_info.precision
        variant = model_info.variant
        repo = model_info.repo

    # Human-friendly description (used by the WebUI)
    precision_descriptions = {
        "fp16": "FP16 (best quality)",
        "fp8_e4m3fn": "FP8 8-bit (good quality)",
        "fp8_e4m3fn_mixed_block35_fp16": "FP8 with last block in FP16 to reduce artifacts (good quality)",
        "Q4_K_M": "GGUF 4-bit quantized (acceptable quality)",
        "Q8_0": "GGUF 8-bit quantized (good quality)",
    }

    if precision:
        description = precision_descriptions.get(precision, precision)
    elif model_name.endswith(".gguf"):
        description = "GGUF quantized"
    elif model_name.endswith(".safetensors"):
        description = "Safetensors"

    if variant == "sharp" or "sharp" in lower:
        description = f"Sharp variant for enhanced detail • {description}" if description else "Sharp variant for enhanced detail"
        if variant is None:
            variant = "sharp"

    return {
        "size": size,
        "precision": precision,
        "variant": variant,
        "repo": repo,
        "description": description,
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    return get_html_template()


@app.get("/api/models")
async def get_models():
    """Get available DiT models."""
    models = get_available_dit_models()
    info = {name: _describe_dit_model(name) for name in models}
    return {"models": models, "default": DEFAULT_DIT_WEBUI, "info": info}


@app.get("/api/status")
async def get_status():
    """Get current system status."""
    gpu_backend = get_gpu_backend()

    gpu_info = "No GPU available"
    if gpu_backend == "cuda":
        if torch.cuda.is_available():
            gpu_info = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory // 1024**3}GB)"
    elif gpu_backend == "mps":
        gpu_info = "Apple Silicon (MPS)"

    return {
        "status": "ready",
        "gpu_backend": gpu_backend,
        "gpu_info": gpu_info,
        "models_cached": len(model_cache) > 0,
        "loaded_model": current_loaded_model
    }


@app.post("/api/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    resolution: int = Form(2160),
    max_resolution: int = Form(4096),
    dit_model: str = Form(DEFAULT_DIT_WEBUI),
    color_correction: str = Form("lab"),
    vae_tiling: bool = Form(True),
    vae_tile_size: int = Form(1024),
    blocks_to_swap: int = Form(0),
    seed: int = Form(42),
    cache_models: bool = Form(True),
    output_format: str = Form("png")
):
    """
    Upscale an image using SeedVR2.

    Args:
        file: Image file to upscale (PNG, JPG, WEBP)
        resolution: Target resolution for shortest edge
        max_resolution: Maximum resolution for any edge (0 = no limit)
        dit_model: DiT model to use
        color_correction: Color correction method
        vae_tiling: Enable VAE tiling for high resolution
        vae_tile_size: Tile size for VAE tiling
        blocks_to_swap: Transformer blocks to swap for VRAM savings
        seed: Random seed for reproducibility
        output_format: Output image format (png, jpeg, jxl)

    Returns:
        JSON with filename and download URL
    """
    # Validate file type
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp", "image/tiff"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    # Validate output format
    if output_format not in ("png", "jpeg", "jxl"):
        output_format = "png"

    try:
        # Read and process image
        image_bytes = await file.read()
        image_tensor = extract_image_tensor(image_bytes)

        # Download models if needed
        model_dir = f"./models/{SEEDVR2_FOLDER_NAME}"
        if not download_weight(dit_model=dit_model, vae_model=DEFAULT_VAE, model_dir=None, debug=debug):
            raise HTTPException(status_code=500, detail="Failed to download required models")

        # Process image
        start_time = time.time()
        result = process_image_upscale(
            image_tensor=image_tensor,
            resolution=resolution,
            max_resolution=max_resolution,
            dit_model=dit_model,
            model_dir=model_dir,
            blocks_to_swap=blocks_to_swap,
            swap_io_components=blocks_to_swap > 0,
            vae_encode_tiled=vae_tiling,
            vae_decode_tiled=vae_tiling,
            vae_tile_size=vae_tile_size,
            color_correction=color_correction,
            seed=seed,
            cache_models=cache_models
        )
        process_time = time.time() - start_time

        # Save to disk
        filename = save_image_to_disk(result, output_format)

        return {
            "filename": filename,
            "url": f"/outputs/{filename}",
            "processing_time": f"{process_time:.2f}s",
            "output_resolution": f"{result.shape[2]}x{result.shape[1]}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upscale_base64")
async def upscale_image_base64(
    image_base64: str = Form(...),
    resolution: int = Form(2160),
    max_resolution: int = Form(4096),
    dit_model: str = Form(DEFAULT_DIT_WEBUI),
    color_correction: str = Form("lab"),
    vae_tiling: bool = Form(True),
    vae_tile_size: int = Form(1024),
    blocks_to_swap: int = Form(0),
    seed: int = Form(42)
):
    """
    Upscale an image from base64 string.

    Returns JSON with base64-encoded result.
    """
    try:
        # Decode base64 image
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        image_bytes = base64.b64decode(image_base64)
        image_tensor = extract_image_tensor(image_bytes)

        # Download models if needed
        model_dir = f"./models/{SEEDVR2_FOLDER_NAME}"
        if not download_weight(dit_model=dit_model, vae_model=DEFAULT_VAE, model_dir=None, debug=debug):
            raise HTTPException(status_code=500, detail="Failed to download required models")

        # Process image
        start_time = time.time()
        result = process_image_upscale(
            image_tensor=image_tensor,
            resolution=resolution,
            max_resolution=max_resolution,
            dit_model=dit_model,
            model_dir=model_dir,
            blocks_to_swap=blocks_to_swap,
            swap_io_components=blocks_to_swap > 0,
            vae_encode_tiled=vae_tiling,
            vae_decode_tiled=vae_tiling,
            vae_tile_size=vae_tile_size,
            color_correction=color_correction,
            seed=seed,
            cache_models=True
        )
        process_time = time.time() - start_time

        # Convert to base64
        output_bytes = tensor_to_png_bytes(result)
        output_base64 = base64.b64encode(output_bytes).decode('utf-8')

        return {
            "success": True,
            "image": f"data:image/png;base64,{output_base64}",
            "processing_time": f"{process_time:.2f}s",
            "output_resolution": f"{result.shape[2]}x{result.shape[1]}"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/api/loaded_model")
async def get_loaded_model():
    """Get the currently loaded model."""
    return {
        "loaded_model": current_loaded_model,
        "models_cached": len(model_cache) > 0
    }


@app.post("/api/clear_cache")
async def clear_cache():
    """Clear model cache and free GPU memory."""
    global model_cache, current_loaded_model
    model_cache = {}
    current_loaded_model = None
    clear_memory(debug=debug, deep=True, force=True)
    return {"status": "success", "message": "Model cache cleared"}

@app.post("/api/clear_outputs")
async def clear_outputs():
    """Clear all generated images from the output directory."""
    try:
        for filename in os.listdir(OUTPUTS_DIR):
            file_path = os.path.join(OUTPUTS_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        return {"status": "success", "message": "Output directory cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HTML Template
# =============================================================================

def get_html_template() -> str:
    """Generate the web UI HTML template."""
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SeedVR2 Image Upscaler v2.1</title>
    <!-- Built: 2024-12-21 Compare Button Fix v3 -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Sora:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a25;
            --bg-hover: #22222f;
            --text-primary: #f0f0f5;
            --text-secondary: #9090a0;
            --text-muted: #606070;
            --accent-primary: #6366f1;
            --accent-secondary: #8b5cf6;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --border-color: rgba(255, 255, 255, 0.08);
            --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
            --font-sans: 'Sora', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html, body {
            height: 100%;
        }

        body {
            font-family: var(--font-sans);
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            overflow-x: hidden;
        }

        /* Background effects */
        .bg-gradient {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background:
                radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.12) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(168, 85, 247, 0.05) 0%, transparent 70%);
            pointer-events: none;
            z-index: -1;
        }

        .noise-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
            opacity: 0.03;
            pointer-events: none;
            z-index: -1;
        }

        /* Main container */
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Header */
        header {
            text-align: center;
            margin-bottom: 3rem;
            animation: fadeInDown 0.6s ease-out;
        }

        .logo {
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }

        .logo-icon {
            width: 48px;
            height: 48px;
            background: var(--accent-gradient);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.4);
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.02em;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 300;
        }

        /* Status bar */
        .status-bar {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 2rem;
            padding: 1rem 1.5rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 2rem;
            font-family: var(--font-mono);
            font-size: 0.85rem;
            animation: fadeIn 0.6s ease-out 0.1s both;
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }

        .status-dot.warning { background: var(--warning); }
        .status-dot.error { background: var(--error); }

        /* Main grid */
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 2rem;
            flex: 1;
            animation: fadeIn 0.6s ease-out 0.2s both;
        }

        @media (max-width: 1024px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Panels */
        .panel {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
        }

        .panel-header {
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .panel-title {
            font-size: 1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .panel-content {
            padding: 1.5rem;
        }

        /* Upload area */
        .upload-zone {
            position: relative;
            min-height: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 2px dashed var(--border-color);
            border-radius: 12px;
            background: var(--bg-tertiary);
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .upload-zone:hover, .upload-zone.dragover {
            border-color: var(--accent-primary);
            background: rgba(99, 102, 241, 0.05);
        }

        .upload-zone.has-image {
            border-style: solid;
            cursor: default;
        }

        .upload-zone.has-image .upload-icon,
        .upload-zone.has-image .upload-text,
        .upload-zone.has-image .upload-hint {
            display: none;
        }

        .upload-icon {
            width: 80px;
            height: 80px;
            margin-bottom: 1.5rem;
            opacity: 0.5;
        }

        .upload-text {
            font-size: 1.1rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .upload-hint {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        #file-input {
            display: none;
        }

        /* Image preview */
        .image-preview {
            display: none;
            position: relative;
            width: 100%;
            height: 100%;
        }

        .image-preview.active {
            display: block;
        }

        .preview-container {
            position: relative;
            width: 100%;
            min-height: 400px;
            aspect-ratio: 16 / 9;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .preview-img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            border-radius: 8px;
        }

        .image-info {
            position: absolute;
            bottom: 1rem;
            left: 1rem;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(8px);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-family: var(--font-mono);
            font-size: 0.8rem;
        }

        .clear-btn {
            position: absolute;
            top: 1rem;
            right: 1rem;
            width: 36px;
            height: 36px;
            background: rgba(0, 0, 0, 0.8);
            border: none;
            border-radius: 50%;
            color: var(--text-primary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }

        .clear-btn:hover {
            background: var(--error);
        }

        /* Settings panel */
        .settings-group {
            margin-bottom: 1.5rem;
        }

        .settings-group:last-child {
            margin-bottom: 0;
        }

        .setting-label {
            display: block;
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            color: var(--text-secondary);
        }

        .setting-row {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        /* Form inputs */
        input[type="number"],
        input[type="text"],
        select {
            width: 100%;
            padding: 0.75rem 1rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-family: var(--font-mono);
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        /* Range slider */
        .range-container {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        input[type="range"] {
            flex: 1;
            -webkit-appearance: none;
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            cursor: pointer;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: var(--accent-primary);
            border-radius: 50%;
            cursor: grab;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4);
        }

        .range-value {
            font-family: var(--font-mono);
            font-size: 0.9rem;
            min-width: 60px;
            text-align: right;
            color: var(--text-secondary);
        }

        /* Toggle switch */
        .toggle-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .toggle {
            position: relative;
            width: 48px;
            height: 26px;
        }

        .toggle input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 26px;
            transition: all 0.3s ease;
        }

        .toggle-slider:before {
            content: "";
            position: absolute;
            height: 18px;
            width: 18px;
            left: 4px;
            bottom: 3px;
            background: var(--text-muted);
            border-radius: 50%;
            transition: all 0.3s ease;
        }

        .toggle input:checked + .toggle-slider {
            background: var(--accent-primary);
            border-color: var(--accent-primary);
        }

        .toggle input:checked + .toggle-slider:before {
            transform: translateX(22px);
            background: white;
        }

        /* Button */
        .btn {
            width: 100%;
            padding: 1rem 1.5rem;
            background: var(--accent-gradient);
            border: none;
            border-radius: 10px;
            color: white;
            font-family: var(--font-sans);
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4);
        }

        .btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.5);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .btn-secondary {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            box-shadow: none;
        }

        .btn-secondary:hover:not(:disabled) {
            background: var(--bg-hover);
            box-shadow: none;
            transform: none;
        }

        /* Progress bar */
        .progress-container {
            display: none;
            margin-top: 1rem;
        }

        .progress-container.active {
            display: block;
        }

        .progress-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .progress-bar {
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: var(--accent-gradient);
            border-radius: 4px;
            width: 0%;
            transition: width 0.3s ease;
        }

        /* Result area */
        .result-panel {
            display: none;
        }

        .result-panel.active {
            display: block;
        }

        .result-actions {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
        }

        .result-actions .btn {
            flex: 1;
        }

        /* Collapsible section */
        .collapsible {
            border-top: 1px solid var(--border-color);
            margin-top: 1.5rem;
            padding-top: 1.5rem;
        }

        .collapsible-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            padding: 0.5rem 0;
        }

        .collapsible-header h3 {
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .collapsible-content {
            display: none;
            margin-top: 1rem;
        }

        .collapsible.open .collapsible-content {
            display: block;
        }

        .collapsible-icon {
            transition: transform 0.3s ease;
        }

        .collapsible.open .collapsible-icon {
            transform: rotate(180deg);
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        /* Comparison slider */
        .comparison-container {
            position: relative;
            width: 100%;
            overflow: hidden;
            border-radius: 8px;
            cursor: ew-resize;
            user-select: none;
        }

        .comparison-container img {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            pointer-events: none;
        }

        .comparison-container img:not([src]),
        .comparison-container img[src=""] {
            display: none;
        }

        .comparison-original {
            z-index: 1;
        }

        .comparison-upscaled {
            z-index: 2;
            clip-path: inset(0 50% 0 0);
        }

        .comparison-slider {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 4px;
            background: white;
            cursor: ew-resize;
            z-index: 10;
            left: 50%;
            transform: translateX(-50%);
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        }

        .comparison-slider::before {
            content: "";
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 44px;
            height: 44px;
            background: white;
            border-radius: 50%;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
        }

        .comparison-slider::after {
            content: "◀ ▶";
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 12px;
            color: var(--bg-primary);
            letter-spacing: 2px;
            white-space: nowrap;
        }

        .comparison-labels {
            position: absolute;
            top: 1rem;
            left: 0;
            right: 0;
            display: flex;
            justify-content: space-between;
            padding: 0 1rem;
            z-index: 5;
            pointer-events: none;
        }

        .comparison-label {
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(8px);
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            font-family: var(--font-mono);
            font-size: 0.75rem;
            color: var(--text-primary);
        }

        .comparison-label.original {
            border: 1px solid var(--error);
        }

        .comparison-label.upscaled {
            border: 1px solid var(--success);
        }

        /* Footer */
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        footer a {
            color: var(--accent-primary);
            text-decoration: none;
        }

        footer a:hover {
            text-decoration: underline;
        }

        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(8px);
            z-index: 1000;
            overflow-y: auto;
            padding: 2rem;
        }

        .modal-overlay.active {
            display: flex;
            align-items: flex-start;
            justify-content: center;
        }

        .modal {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            width: 100%;
            max-width: 900px;
            margin: 2rem 0;
            animation: fadeInDown 0.3s ease-out;
        }

        .modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }

        .modal-title {
            font-size: 1.25rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .modal-close {
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 0.5rem;
            border-radius: 8px;
            transition: all 0.2s;
        }

        .modal-close:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }

        .modal-body {
            padding: 1.5rem;
            max-height: 70vh;
            overflow-y: auto;
        }

        /* API Docs styling */
        .api-section {
            margin-bottom: 2rem;
        }

        .api-section:last-child {
            margin-bottom: 0;
        }

        .api-endpoint {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 1rem;
            overflow: hidden;
        }

        .api-endpoint-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem 1.25rem;
            cursor: pointer;
            transition: background 0.2s;
        }

        .api-endpoint-header:hover {
            background: var(--bg-hover);
        }

        .api-method {
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            text-transform: uppercase;
        }

        .api-method.get {
            background: rgba(34, 197, 94, 0.2);
            color: var(--success);
        }

        .api-method.post {
            background: rgba(99, 102, 241, 0.2);
            color: var(--accent-primary);
        }

        .api-path {
            font-family: var(--font-mono);
            font-size: 0.9rem;
            color: var(--text-primary);
        }

        .api-desc {
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-left: auto;
        }

        .api-endpoint-body {
            display: none;
            padding: 1.25rem;
            border-top: 1px solid var(--border-color);
            background: rgba(0, 0, 0, 0.2);
        }

        .api-endpoint.open .api-endpoint-body {
            display: block;
        }

        .api-param-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }

        .api-param-table th,
        .api-param-table td {
            text-align: left;
            padding: 0.75rem;
            border-bottom: 1px solid var(--border-color);
        }

        .api-param-table th {
            color: var(--text-secondary);
            font-weight: 500;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .api-param-name {
            font-family: var(--font-mono);
            color: var(--accent-primary);
        }

        .api-param-type {
            font-family: var(--font-mono);
            color: var(--text-muted);
            font-size: 0.8rem;
        }

        .api-param-required {
            color: var(--error);
            font-size: 0.75rem;
        }

        .code-block {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            font-family: var(--font-mono);
            font-size: 0.8rem;
            overflow-x: auto;
            white-space: pre;
            color: var(--text-secondary);
        }

        .code-block .keyword { color: #c678dd; }
        .code-block .string { color: #98c379; }
        .code-block .number { color: #d19a66; }
        .code-block .comment { color: #5c6370; }

        h4 {
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: var(--text-secondary);
        }
    </style>
</head>
<body>
    <div class="bg-gradient"></div>
    <div class="noise-overlay"></div>

    <div class="container">
        <header>
            <div class="logo">
                <div class="logo-icon">✨</div>
                <h1>SeedVR2 Upscaler</h1>
            </div>
            <p class="subtitle">AI-powered image upscaling with diffusion models</p>
        </header>

        <div class="status-bar">
            <div class="status-item">
                <span class="status-dot" id="status-dot"></span>
                <span id="status-text">Loading...</span>
            </div>
            <div class="status-item" id="gpu-info">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                    <line x1="8" y1="21" x2="16" y2="21"></line>
                    <line x1="12" y1="17" x2="12" y2="21"></line>
                </svg>
                <span id="gpu-text">Detecting GPU...</span>
            </div>
            <div class="status-item" id="model-status" style="display: none;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                    <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                    <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
                <span id="loaded-model-text">No model</span>
            </div>
        </div>

        <div class="main-grid">
            <!-- Image Panel -->
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                            <circle cx="8.5" cy="8.5" r="1.5"></circle>
                            <polyline points="21 15 16 10 5 21"></polyline>
                        </svg>
                        Image
                    </span>
                    <button class="btn btn-secondary" id="compare-btn" style="display: none; padding: 0.5rem 1rem; width: auto;">
                        Compare
                    </button>
                </div>
                <div class="panel-content">
                    <div class="upload-zone" id="upload-zone">
                        <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        <p class="upload-text">Drop an image here or click to upload</p>
                        <p class="upload-hint">Supports PNG, JPG, WEBP • Max 50MB</p>
                        <input type="file" id="file-input" accept="image/*">

                        <div class="image-preview" id="image-preview">
                            <!-- Standard preview mode -->
                            <div class="preview-container" id="preview-container">
                                <img src="" alt="Preview" class="preview-img" id="preview-img">
                            </div>

                            <!-- Comparison mode -->
                            <div class="preview-container comparison-container" id="comparison-container" style="display: none;">
                                <img src="" alt="Original" class="comparison-original" id="comparison-original">
                                <img src="" alt="Upscaled" class="comparison-upscaled" id="comparison-upscaled">
                                <div class="comparison-slider" id="comparison-slider"></div>
                                <div class="comparison-labels">
                                    <span class="comparison-label original">Original</span>
                                    <span class="comparison-label upscaled">Upscaled</span>
                                </div>
                            </div>

                            <div class="image-info" id="image-info"></div>
                            <button class="clear-btn" id="clear-btn">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <line x1="18" y1="6" x2="6" y2="18"></line>
                                    <line x1="6" y1="6" x2="18" y2="18"></line>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Settings Panel -->
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="3"></circle>
                            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                        </svg>
                        Settings
                    </span>
                </div>
                <div class="panel-content">
                    <div class="settings-group">
                        <label class="setting-label">Target Resolution (shortest edge)</label>
                        <div class="range-container">
                            <input type="range" id="resolution" min="480" max="2160" value="2160" step="120">
                            <span class="range-value" id="resolution-value">2160px</span>
                        </div>
                    </div>

                    <div class="settings-group">
                        <label class="setting-label">Max Resolution (0 = no limit)</label>
                        <div class="range-container">
                            <input type="range" id="max-resolution" min="0" max="4096" value="4096" step="256">
                            <span class="range-value" id="max-resolution-value">4096px</span>
                        </div>
                    </div>

                    <div class="settings-group">
                        <label class="setting-label">Color Correction</label>
                        <select id="color-correction">
                            <option value="lab" selected>LAB (Recommended)</option>
                            <option value="wavelet">Wavelet</option>
                            <option value="wavelet_adaptive">Wavelet Adaptive</option>
                            <option value="hsv">HSV</option>
                            <option value="adain">AdaIN</option>
                            <option value="none">None</option>
                        </select>
                    </div>

                    <div class="settings-group">
                        <label class="setting-label">Output Format</label>
                        <select id="output-format">
                            <option value="png" selected>PNG</option>
                            <option value="jpeg">JPEG</option>
                            <option value="jxl">JPEG XL</option>
                        </select>
                    </div>

                    <div class="settings-group">
                        <label class="setting-label">Model</label>
                        <select id="dit-model">
                            <option value="">Loading models...</option>
                        </select>
                        <div id="model-info" style="margin-top: 0.5rem; font-size: 0.75rem; color: var(--text-muted); line-height: 1.4;">
                            Loading model info...
                        </div>
                        <details style="margin-top: 0.75rem;">
                            <summary style="cursor: pointer; font-size: 0.8rem; color: var(--text-secondary);">
                                Show model details
                            </summary>
                            <div id="model-details" style="margin-top: 0.5rem; font-size: 0.75rem; color: var(--text-muted); line-height: 1.4;"></div>
                        </details>
                    </div>

                    <div class="settings-group">
                        <label class="setting-label">Seed</label>
                        <input type="number" id="seed" value="42" min="0" max="999999999">
                    </div>

                    <div class="collapsible" id="advanced-settings">
                        <div class="collapsible-header">
                            <h3>Advanced Settings</h3>
                            <svg class="collapsible-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="6 9 12 15 18 9"></polyline>
                            </svg>
                        </div>
                        <div class="collapsible-content">
                            <div class="settings-group">
                                <div class="toggle-container">
                                    <span class="setting-label" style="margin-bottom: 0;">VAE Tiling</span>
                                    <label class="toggle">
                                        <input type="checkbox" id="vae-tiling" checked>
                                        <span class="toggle-slider"></span>
                                    </label>
                                </div>
                                <p style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">
                                    Enable for high resolution outputs (reduces VRAM usage)
                                </p>
                            </div>

                            <div class="settings-group">
                                <label class="setting-label">VAE Tile Size</label>
                                <input type="number" id="vae-tile-size" value="1024" min="256" max="2048" step="64">
                            </div>

                            <div class="settings-group">
                                <label class="setting-label">BlockSwap (VRAM savings)</label>
                                <div class="range-container">
                                    <input type="range" id="blocks-to-swap" min="0" max="32" value="0" step="4">
                                    <span class="range-value" id="blocks-value">0</span>
                                </div>
                                <p style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">
                                    Higher = less VRAM but slower. Use for 8GB GPUs.
                                </p>
                            </div>

                            <div class="settings-group">
                                <div class="toggle-container">
                                    <span class="setting-label" style="margin-bottom: 0;">Keep Models in VRAM</span>
                                    <label class="toggle">
                                        <input type="checkbox" id="cache-models" checked>
                                        <span class="toggle-slider"></span>
                                    </label>
                                </div>
                                <p style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">
                                    Faster repeated upscaling (uses ~8GB VRAM)
                                </p>
                            </div>

                            <div class="settings-group" style="margin-top: 1rem;">
                                <button class="btn btn-secondary" id="clear-outputs-btn" style="font-size: 0.9rem; padding: 0.5rem 1rem;">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 0.5rem;">
                                        <polyline points="3 6 5 6 21 6"></polyline>
                                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                                    </svg>
                                    Clear Generated Images
                                </button>
                            </div>
                        </div>
                    </div>

                    <button class="btn" id="upscale-btn" disabled>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
                            <polyline points="17 6 23 6 23 12"></polyline>
                        </svg>
                        Upscale Image
                    </button>

                    <div class="progress-container" id="progress-container">
                        <div class="progress-label">
                            <span id="progress-phase">Preparing...</span>
                            <span id="progress-percent">0%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="progress-fill"></div>
                        </div>
                    </div>

                    <div class="result-panel" id="result-panel">
                        <div class="result-actions">
                            <button class="btn" id="download-btn">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                    <polyline points="7 10 12 15 17 10"></polyline>
                                    <line x1="12" y1="15" x2="12" y2="3"></line>
                                </svg>
                                Download
                            </button>
                            <button class="btn btn-secondary" id="new-btn">
                                New Image
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            Powered by <a href="https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler" target="_blank">SeedVR2</a> •
            AI upscaling with diffusion models •
            <a href="#" id="api-docs-link">API Documentation</a>
        </footer>
    </div>

    <!-- API Documentation Modal -->
    <div class="modal-overlay" id="api-modal">
        <div class="modal">
            <div class="modal-header">
                <span class="modal-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="16" y1="13" x2="8" y2="13"></line>
                        <line x1="16" y1="17" x2="8" y2="17"></line>
                        <polyline points="10 9 9 9 8 9"></polyline>
                    </svg>
                    API Documentation
                </span>
                <button class="modal-close" id="api-modal-close">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            </div>
            <div class="modal-body">
                <div class="api-section">
                    <h3 style="margin-bottom: 1rem; color: var(--text-secondary);">Endpoints</h3>

                    <!-- POST /api/upscale -->
                    <div class="api-endpoint" id="endpoint-upscale">
                        <div class="api-endpoint-header" onclick="toggleEndpoint('endpoint-upscale')">
                            <span class="api-method post">POST</span>
                            <span class="api-path">/api/upscale</span>
                            <span class="api-desc">Upscale an image file</span>
                        </div>
                        <div class="api-endpoint-body">
                            <h4>Parameters (multipart/form-data)</h4>
                            <table class="api-param-table">
                                <thead>
                                    <tr>
                                        <th>Name</th>
                                        <th>Type</th>
                                        <th>Default</th>
                                        <th>Description</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td><span class="api-param-name">file</span> <span class="api-param-required">*</span></td>
                                        <td><span class="api-param-type">File</span></td>
                                        <td>—</td>
                                        <td>Image file (PNG, JPG, WEBP)</td>
                                    </tr>
                                    <tr>
                                        <td><span class="api-param-name">resolution</span></td>
                                        <td><span class="api-param-type">int</span></td>
                                        <td>2160</td>
                                        <td>Target resolution for shortest edge</td>
                                    </tr>
                                    <tr>
                                        <td><span class="api-param-name">max_resolution</span></td>
                                        <td><span class="api-param-type">int</span></td>
                                        <td>4096</td>
                                        <td>Max resolution for any edge (0 = no limit)</td>
                                    </tr>
                                    <tr>
                                        <td><span class="api-param-name">dit_model</span></td>
                                        <td><span class="api-param-type">string</span></td>
                                        <td>seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors</td>
                                        <td>Model to use for upscaling</td>
                                    </tr>
                                    <tr>
                                        <td><span class="api-param-name">color_correction</span></td>
                                        <td><span class="api-param-type">string</span></td>
                                        <td>lab</td>
                                        <td>lab, wavelet, hsv, adain, none</td>
                                    </tr>
                                    <tr>
                                        <td><span class="api-param-name">vae_tiling</span></td>
                                        <td><span class="api-param-type">bool</span></td>
                                        <td>true</td>
                                        <td>Enable VAE tiling for high resolution</td>
                                    </tr>
                                    <tr>
                                        <td><span class="api-param-name">seed</span></td>
                                        <td><span class="api-param-type">int</span></td>
                                        <td>42</td>
                                        <td>Random seed for reproducibility</td>
                                    </tr>
                                    <tr>
                                        <td><span class="api-param-name">cache_models</span></td>
                                        <td><span class="api-param-type">bool</span></td>
                                        <td>true</td>
                                        <td>Keep models in VRAM for faster inference</td>
                                    </tr>
                                    <tr>
                                        <td><span class="api-param-name">output_format</span></td>
                                        <td><span class="api-param-type">string</span></td>
                                        <td>png</td>
                                        <td>Output format (png, jpeg, jxl)</td>
                                    </tr>
                                </tbody>
                            </table>

                            <h4>Response (JSON)</h4>
                            <div class="code-block">{
    "filename": "upscaled_xyz.png",
    "url": "/outputs/upscaled_xyz.png",
    "processing_time": "5.23s",
    "output_resolution": "1920x1080"
}</div>
                        </div>
                    </div>

                    <!-- POST /api/upscale_base64 -->
                    <div class="api-endpoint" id="endpoint-base64">
                        <div class="api-endpoint-header" onclick="toggleEndpoint('endpoint-base64')">
                            <span class="api-method post">POST</span>
                            <span class="api-path">/api/upscale_base64</span>
                            <span class="api-desc">Upscale a base64-encoded image</span>
                        </div>
                        <div class="api-endpoint-body">
                            <h4>Parameters (form-data)</h4>
                            <table class="api-param-table">
                                <thead>
                                    <tr>
                                        <th>Name</th>
                                        <th>Type</th>
                                        <th>Description</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td><span class="api-param-name">image_base64</span> <span class="api-param-required">*</span></td>
                                        <td><span class="api-param-type">string</span></td>
                                        <td>Base64-encoded image (with or without data URI prefix)</td>
                                    </tr>
                                    <tr>
                                        <td colspan="3" style="color: var(--text-muted);">...same parameters as /api/upscale</td>
                                    </tr>
                                </tbody>
                            </table>

                            <h4>Response (JSON)</h4>
                            <div class="code-block">{
  <span class="string">"success"</span>: <span class="keyword">true</span>,
  <span class="string">"image"</span>: <span class="string">"data:image/png;base64,iVBORw0KGg..."</span>,
  <span class="string">"processing_time"</span>: <span class="string">"5.23s"</span>,
  <span class="string">"output_resolution"</span>: <span class="string">"1920x1080"</span>
}</div>
                        </div>
                    </div>

                    <!-- GET /api/models -->
                    <div class="api-endpoint" id="endpoint-models">
                        <div class="api-endpoint-header" onclick="toggleEndpoint('endpoint-models')">
                            <span class="api-method get">GET</span>
                            <span class="api-path">/api/models</span>
                            <span class="api-desc">List available models</span>
                        </div>
                        <div class="api-endpoint-body">
                            <h4>Response (JSON)</h4>
                            <div class="code-block">{
  <span class="string">"models"</span>: [
    <span class="string">"seedvr2_ema_3b_fp8_e4m3fn.safetensors"</span>,
    <span class="string">"seedvr2_ema_3b_fp16.safetensors"</span>,
    <span class="string">"seedvr2_ema_7b_fp8_e4m3fn.safetensors"</span>,
    ...
  ],
  <span class="string">"default"</span>: <span class="string">"seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"</span>
}</div>
                        </div>
                    </div>

                    <!-- GET /api/status -->
                    <div class="api-endpoint" id="endpoint-status">
                        <div class="api-endpoint-header" onclick="toggleEndpoint('endpoint-status')">
                            <span class="api-method get">GET</span>
                            <span class="api-path">/api/status</span>
                            <span class="api-desc">Get system status</span>
                        </div>
                        <div class="api-endpoint-body">
                            <h4>Response (JSON)</h4>
                            <div class="code-block">{
  <span class="string">"status"</span>: <span class="string">"ready"</span>,
  <span class="string">"gpu_backend"</span>: <span class="string">"cuda"</span>,
  <span class="string">"gpu_info"</span>: <span class="string">"NVIDIA RTX 4090 (24GB)"</span>,
  <span class="string">"models_cached"</span>: <span class="keyword">true</span>
}</div>
                        </div>
                    </div>

                    <!-- POST /api/clear_cache -->
                    <div class="api-endpoint" id="endpoint-cache">
                        <div class="api-endpoint-header" onclick="toggleEndpoint('endpoint-cache')">
                            <span class="api-method post">POST</span>
                            <span class="api-path">/api/clear_cache</span>
                            <span class="api-desc">Clear model cache and free VRAM</span>
                        </div>
                        <div class="api-endpoint-body">
                            <h4>Response (JSON)</h4>
                            <div class="code-block">{
  <span class="string">"status"</span>: <span class="string">"success"</span>,
  <span class="string">"message"</span>: <span class="string">"Model cache cleared"</span>
}</div>
                        </div>
                    </div>
                </div>

                <div class="api-section">
                    <h3 style="margin-bottom: 1rem; color: var(--text-secondary);">OpenAPI / Swagger</h3>
                    <p style="font-size: 0.9rem; color: var(--text-secondary);">
                        Interactive API documentation is available at
                        <a href="/docs" target="_blank">/docs</a> (Swagger UI) or
                        <a href="/redoc" target="_blank">/redoc</a> (ReDoc).
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // State
        let currentFile = null;
        let resultUrl = null;
        let resultFilename = null;
        let isProcessing = false;

        // DOM elements
        const uploadZone = document.getElementById('upload-zone');
        const fileInput = document.getElementById('file-input');
        const imagePreview = document.getElementById('image-preview');
        const previewImg = document.getElementById('preview-img');
        const imageInfo = document.getElementById('image-info');
        const clearBtn = document.getElementById('clear-btn');
        const upscaleBtn = document.getElementById('upscale-btn');
        const progressContainer = document.getElementById('progress-container');
        const progressPhase = document.getElementById('progress-phase');
        const progressPercent = document.getElementById('progress-percent');
        const progressFill = document.getElementById('progress-fill');
        const resultPanel = document.getElementById('result-panel');
        const downloadBtn = document.getElementById('download-btn');
        const newBtn = document.getElementById('new-btn');
        const compareBtn = document.getElementById('compare-btn');
        const ditModelSelect = document.getElementById('dit-model');
        const resolutionSlider = document.getElementById('resolution');
        const resolutionValue = document.getElementById('resolution-value');
        const maxResolutionSlider = document.getElementById('max-resolution');
        const maxResolutionValue = document.getElementById('max-resolution-value');
        const blocksSlider = document.getElementById('blocks-to-swap');
        const blocksValue = document.getElementById('blocks-value');
        const advancedSettings = document.getElementById('advanced-settings');
        const modelInfoText = document.getElementById('model-info');
        const modelDetails = document.getElementById('model-details');
        const clearOutputsBtn = document.getElementById('clear-outputs-btn');
        const outputFormatSelect = document.getElementById('output-format');

        // Model metadata (populated from /api/models)
        let modelInfoMap = {};

        function escapeHtml(str) {
            return (str || '')
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }

        function getShortName(model) {
            return model
                .replace('seedvr2_ema_', '')
                .replace('.safetensors', '')
                .replace('.gguf', ' (GGUF)');
        }

        function updateModelInfo() {
            if (!modelInfoText) return;

            const model = ditModelSelect.value;
            if (!model) {
                modelInfoText.textContent = '';
                return;
            }

            const info = modelInfoMap[model];
            const parts = [];
            if (info && info.size) parts.push(info.size);
            if (info && info.description) parts.push(info.description);

            modelInfoText.textContent = parts.length > 0
                ? parts.join(' • ')
                : 'No metadata available for this model.';
        }

        function renderModelDetails(models) {
            if (!modelDetails) return;
            if (!models || models.length === 0) {
                modelDetails.textContent = 'No models found.';
                return;
            }

            const groups = { "7B": [], "3B": [], "Other": [] };

            models.forEach(m => {
                const info = modelInfoMap[m] || {};
                const lower = (m || '').toLowerCase();
                const size = info.size || (lower.includes('7b') ? '7B' : (lower.includes('3b') ? '3B' : 'Other'));
                (groups[size] || groups.Other).push(m);
            });

            const order = ['7B', '3B', 'Other'];
            let html = '';

            order.forEach(size => {
                const list = groups[size] || [];
                if (list.length === 0) return;

                const title = size === 'Other' ? 'Other models' : (size + ' models');
                html +=
                    '<div style="margin-top: 0.5rem;">' +
                        '<div style="font-weight: 600; color: var(--text-secondary); margin-bottom: 0.25rem;">' + title + '</div>' +
                        '<ul style="margin: 0; padding-left: 1.1rem;">' +
                            list.map(m => {
                                const info = modelInfoMap[m] || {};
                                const desc = info.description ? (' — ' + escapeHtml(info.description)) : '';
                                return '<li><span style="font-family: var(--font-mono);">' + escapeHtml(m) + '</span>' + desc + '</li>';
                            }).join('') +
                        '</ul>' +
                    '</div>';
            });

            modelDetails.innerHTML = html;
        }

        // Initialize
        async function init() {
            await updateStatus();

            // Fetch models
            try {
                const response = await fetch('/api/models');
                const data = await response.json();

                modelInfoMap = data.info || {};

                ditModelSelect.innerHTML = data.models.map(model => {
                    // Shorten model names for display
                    const shortName = getShortName(model);
                    const info = modelInfoMap[model];
                    const label = (info && info.size) ? ('[' + info.size + '] ' + shortName) : shortName;
                    return '<option value="' + model + '"' + (model === data.default ? ' selected' : '') + '>' + label + '</option>';
                }).join('');

                updateModelInfo();
                renderModelDetails(data.models);
            } catch (e) {
                console.error('Failed to fetch models:', e);
            }
        }

        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();

                document.getElementById('status-dot').className = 'status-dot';
                document.getElementById('status-text').textContent = status.status === 'ready' ? 'Ready' : 'Loading...';
                document.getElementById('gpu-text').textContent = status.gpu_info;

                // Show loaded model
                const modelStatus = document.getElementById('model-status');
                const loadedModelText = document.getElementById('loaded-model-text');

                if (status.loaded_model) {
                    const shortName = status.loaded_model.replace('seedvr2_ema_', '').replace('.safetensors', '').replace('.gguf', ' (GGUF)');
                    loadedModelText.textContent = shortName;
                    modelStatus.style.display = 'flex';
                } else {
                    modelStatus.style.display = 'none';
                }
            } catch (e) {
                document.getElementById('status-dot').className = 'status-dot error';
                document.getElementById('status-text').textContent = 'Error';
            }
        }

        // Event listeners
        ditModelSelect.addEventListener('change', updateModelInfo);

        uploadZone.addEventListener('click', (e) => {
            if (!imagePreview.classList.contains('active')) {
                fileInput.click();
            }
        });

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');

            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        clearBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetUI();
        });

        upscaleBtn.addEventListener('click', () => {
            if (!isProcessing && currentFile) {
                processImage();
            }
        });

        downloadBtn.addEventListener('click', () => {
            if (resultUrl) {
                const a = document.createElement('a');
                a.href = resultUrl;
                a.download = resultFilename || 'upscaled.png';
                a.click();
            }
        });

        newBtn.addEventListener('click', resetUI);

        clearOutputsBtn.addEventListener('click', async () => {
            if (!confirm('Are you sure you want to delete all generated images? This cannot be undone.')) return;

            try {
                const response = await fetch('/api/clear_outputs', { method: 'POST' });
                const data = await response.json();
                if (response.ok) {
                    alert('Output history cleared successfully.');
                } else {
                    alert('Failed to clear output history: ' + (data.detail || 'Unknown error'));
                }
            } catch (e) {
                alert('Error clearing output history: ' + e.message);
            }
        });

        // Comparison slider functionality
        const comparisonContainer = document.getElementById('comparison-container');
        const previewContainer = document.getElementById('preview-container');
        const comparisonOriginal = document.getElementById('comparison-original');
        const comparisonUpscaled = document.getElementById('comparison-upscaled');
        const comparisonSlider = document.getElementById('comparison-slider');
        let isComparing = false;
        let isDragging = false;

        // Keep the frame height dynamic by matching the current image aspect ratio.
        function setFrameAspect(width, height) {
            if (!width || !height) return;
            const ratio = width + ' / ' + height;
            previewContainer.style.aspectRatio = ratio;
            comparisonContainer.style.aspectRatio = ratio;
        }

        previewImg.addEventListener('load', () => {
            setFrameAspect(previewImg.naturalWidth, previewImg.naturalHeight);
        });

        comparisonOriginal.addEventListener('load', () => {
            setFrameAspect(comparisonOriginal.naturalWidth, comparisonOriginal.naturalHeight);
        });

        comparisonUpscaled.addEventListener('load', () => {
            setFrameAspect(comparisonUpscaled.naturalWidth, comparisonUpscaled.naturalHeight);
        });

        compareBtn.addEventListener('click', () => {
            // Only allow comparison if we have both original and upscaled images
            if (!resultUrl || !previewImg.dataset.originalSrc) {
                return;
            }

            if (!isComparing) {
                // Ensure comparison images are always up-to-date
                comparisonOriginal.src = previewImg.dataset.originalSrc;
                comparisonUpscaled.src = previewImg.src;

                // Switch to comparison mode
                previewContainer.style.display = 'none';
                comparisonContainer.style.display = 'flex';
                compareBtn.textContent = 'Exit Compare';
                isComparing = true;
                updateSliderPosition(50);
            } else {
                // Switch back to result view
                comparisonContainer.style.display = 'none';
                previewContainer.style.display = 'flex';
                previewImg.src = resultUrl;
                compareBtn.textContent = 'Compare';
                isComparing = false;
            }
        });

        function updateSliderPosition(percentage) {
            percentage = Math.max(0, Math.min(100, percentage));
            comparisonSlider.style.left = percentage + '%';
            comparisonUpscaled.style.clipPath = 'inset(0 ' + (100 - percentage) + '% 0 0)';
        }

        comparisonContainer.addEventListener('mousedown', (e) => {
            isDragging = true;
            updateSliderFromEvent(e);
        });

        comparisonContainer.addEventListener('touchstart', (e) => {
            isDragging = true;
            updateSliderFromEvent(e.touches[0]);
        });

        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                updateSliderFromEvent(e);
            }
        });

        document.addEventListener('touchmove', (e) => {
            if (isDragging) {
                updateSliderFromEvent(e.touches[0]);
            }
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
        });

        document.addEventListener('touchend', () => {
            isDragging = false;
        });

        function updateSliderFromEvent(e) {
            const rect = comparisonContainer.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const percentage = (x / rect.width) * 100;
            updateSliderPosition(percentage);
        }

        advancedSettings.querySelector('.collapsible-header').addEventListener('click', () => {
            advancedSettings.classList.toggle('open');
        });

        resolutionSlider.addEventListener('input', () => {
            resolutionValue.textContent = resolutionSlider.value + 'px';
        });

        maxResolutionSlider.addEventListener('input', () => {
            const val = parseInt(maxResolutionSlider.value);
            maxResolutionValue.textContent = val === 0 ? 'None' : val + 'px';
        });

        blocksSlider.addEventListener('input', () => {
            blocksValue.textContent = blocksSlider.value;
        });

        // Functions
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please upload an image file');
                return;
            }

            if (file.size > 50 * 1024 * 1024) {
                alert('File too large. Maximum size is 50MB.');
                return;
            }

            currentFile = file;
            resultUrl = null;  // Clear any previous result when new file is uploaded
            resultFilename = null;

            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    previewImg.src = e.target.result;
                    previewImg.dataset.originalSrc = e.target.result;  // Store for comparison
                    setFrameAspect(img.width, img.height);
                    imageInfo.textContent = img.width + ' × ' + img.height + 'px • ' + (file.size / 1024 / 1024).toFixed(2) + 'MB';

                    imagePreview.classList.add('active');
                    uploadZone.classList.add('has-image');
                    upscaleBtn.disabled = false;
                    resultPanel.classList.remove('active');

                    // Reset compare button
                    compareBtn.style.display = 'none';
                    compareBtn.textContent = 'Compare';

                    // Reset comparison state
                    previewContainer.style.display = 'flex';
                    comparisonContainer.style.display = 'none';
                    comparisonOriginal.src = '';
                    comparisonUpscaled.src = '';
                    isComparing = false;
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        function resetUI() {
            currentFile = null;
            resultUrl = null;
            resultFilename = null;
            fileInput.value = '';
            previewImg.src = '';
            previewImg.dataset.originalSrc = '';
            imageInfo.textContent = '';
            imagePreview.classList.remove('active');
            uploadZone.classList.remove('has-image');
            upscaleBtn.disabled = true;
            progressContainer.classList.remove('active');
            resultPanel.classList.remove('active');

            // Reset frame aspect ratio to default
            previewContainer.style.aspectRatio = '';
            comparisonContainer.style.aspectRatio = '';

            // Reset compare button
            compareBtn.style.display = 'none';
            compareBtn.textContent = 'Compare';

            // Reset comparison
            comparisonOriginal.src = '';
            comparisonUpscaled.src = '';
            comparisonContainer.style.display = 'none';
            previewContainer.style.display = 'flex';
            isComparing = false;
        }

        async function processImage() {
            if (!currentFile || isProcessing) return;

            isProcessing = true;
            upscaleBtn.disabled = true;
            upscaleBtn.innerHTML = '<span class="spinner"></span> Processing...';
            progressContainer.classList.add('active');

            // Simulate progress phases
            const phases = [
                { name: 'Downloading models...', duration: 500 },
                { name: 'Encoding...', progress: 25 },
                { name: 'Upscaling...', progress: 50 },
                { name: 'Decoding...', progress: 75 },
                { name: 'Post-processing...', progress: 90 }
            ];

            let phaseIndex = 0;
            const updateProgress = () => {
                if (phaseIndex < phases.length) {
                    const phase = phases[phaseIndex];
                    progressPhase.textContent = phase.name;
                    if (phase.progress) {
                        progressPercent.textContent = phase.progress + '%';
                        progressFill.style.width = phase.progress + '%';
                    }
                    phaseIndex++;
                    setTimeout(updateProgress, 2000);
                }
            };
            updateProgress();

            try {
                const formData = new FormData();
                formData.append('file', currentFile);
                formData.append('resolution', resolutionSlider.value);
                formData.append('max_resolution', maxResolutionSlider.value);
                formData.append('dit_model', ditModelSelect.value);
                formData.append('color_correction', document.getElementById('color-correction').value);
                formData.append('vae_tiling', document.getElementById('vae-tiling').checked);
                formData.append('vae_tile_size', document.getElementById('vae-tile-size').value);
                formData.append('blocks_to_swap', blocksSlider.value);
                formData.append('seed', document.getElementById('seed').value);
                formData.append('cache_models', document.getElementById('cache-models').checked);
                formData.append('output_format', outputFormatSelect.value);

                const response = await fetch('/api/upscale', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Upscaling failed');
                }

                const data = await response.json();
                resultUrl = data.url;
                resultFilename = data.filename;

                const processingTime = data.processing_time;
                const outputRes = data.output_resolution;

                // Update frame aspect ratio early (before image load) to avoid side gutters
                if (outputRes) {
                    const m = outputRes.match(/(\d+)\s*[x×]\s*(\d+)/i);
                    if (m) {
                        setFrameAspect(parseInt(m[1], 10), parseInt(m[2], 10));
                    }
                }

                // Show result
                previewImg.src = resultUrl;

                // Setup comparison images
                comparisonOriginal.src = previewImg.dataset.originalSrc || previewImg.src;
                comparisonUpscaled.src = resultUrl;

                imageInfo.textContent = outputRes + ' • ' + processingTime;

                progressPercent.textContent = '100%';
                progressFill.style.width = '100%';
                progressPhase.textContent = 'Complete!';

                resultPanel.classList.add('active');
                compareBtn.style.display = 'block';
                compareBtn.textContent = 'Compare';
                isComparing = false;

                // Update status to show loaded model
                updateStatus();

            } catch (error) {
                alert('Error: ' + error.message);
                progressContainer.classList.remove('active');
            } finally {
                isProcessing = false;
                upscaleBtn.disabled = false;
                upscaleBtn.innerHTML = `
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
                        <polyline points="17 6 23 6 23 12"></polyline>
                    </svg>
                    Upscale Image
                `;
            }
        }

        // API Documentation Modal
        const apiModal = document.getElementById('api-modal');
        const apiDocsLink = document.getElementById('api-docs-link');
        const apiModalClose = document.getElementById('api-modal-close');

        apiDocsLink.addEventListener('click', (e) => {
            e.preventDefault();
            apiModal.classList.add('active');
            document.body.style.overflow = 'hidden';
        });

        apiModalClose.addEventListener('click', () => {
            apiModal.classList.remove('active');
            document.body.style.overflow = '';
        });

        apiModal.addEventListener('click', (e) => {
            if (e.target === apiModal) {
                apiModal.classList.remove('active');
                document.body.style.overflow = '';
            }
        });

        function toggleEndpoint(id) {
            const endpoint = document.getElementById(id);
            endpoint.classList.toggle('open');
        }

        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && apiModal.classList.contains('active')) {
                apiModal.classList.remove('active');
                document.body.style.overflow = '';
            }
        });

        // Ensure clean initial UI state on page load
        function ensureCleanState() {
            // Reset all state variables
            currentFile = null;
            resultUrl = null;
            resultFilename = null;
            isProcessing = false;

            // Hide compare button until we have a result
            compareBtn.style.display = 'none';
            compareBtn.textContent = 'Compare';

            // Reset frame aspect ratio to default
            previewContainer.style.aspectRatio = '';
            comparisonContainer.style.aspectRatio = '';

            // Ensure not in comparison mode
            comparisonContainer.style.display = 'none';
            comparisonContainer.style.display = 'none';
            previewContainer.style.display = 'flex';

            // Reset image preview
            imagePreview.classList.remove('active');
            uploadZone.classList.remove('has-image');
            previewImg.src = '';
            previewImg.dataset.originalSrc = '';
            imageInfo.textContent = '';

            // Reset comparison images
            comparisonOriginal.src = '';
            comparisonUpscaled.src = '';

            // Reset result panel
            resultPanel.classList.remove('active');
            progressContainer.classList.remove('active');
            upscaleBtn.disabled = true;
        }

        // Handle bfcache (back-forward cache) restoration
        window.addEventListener('pageshow', (event) => {
            if (event.persisted) {
                // Page was restored from bfcache, reset state
                ensureCleanState();
                init();
            }
        });

        // Initialize on load
        ensureCleanState();
        init();
    </script>
</body>
</html>
"""


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SeedVR2 Image Upscaler - Web UI and API",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind to (default: 8000)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload for development")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Update debug instance
    debug.enabled = args.debug

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   SeedVR2 Image Upscaler                     ║
║                      Web UI & API                            ║
╠══════════════════════════════════════════════════════════════╣
║  Local URL:    http://{args.host}:{args.port:<23}       ║
║  API Docs:     http://{args.host}:{args.port}/docs{' ' * 16}       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "webui:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug" if args.debug else "info"
    )


if __name__ == "__main__":
    main()
