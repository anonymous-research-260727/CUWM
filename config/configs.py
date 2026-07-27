import os
from diffsynth.pipelines.qwen_image import ModelConfig

MODEL_BASE = os.environ.get("MODEL_BASE", "models/Qwen")

MODEL_CONFIGS = [
    ModelConfig(
        path=[
            os.path.join(MODEL_BASE, "Qwen-Image-Edit-2509/transformer/diffusion_pytorch_model-00001-of-00005.safetensors"),
            os.path.join(MODEL_BASE, "Qwen-Image-Edit-2509/transformer/diffusion_pytorch_model-00002-of-00005.safetensors"),
            os.path.join(MODEL_BASE, "Qwen-Image-Edit-2509/transformer/diffusion_pytorch_model-00003-of-00005.safetensors"),
            os.path.join(MODEL_BASE, "Qwen-Image-Edit-2509/transformer/diffusion_pytorch_model-00004-of-00005.safetensors"),
            os.path.join(MODEL_BASE, "Qwen-Image-Edit-2509/transformer/diffusion_pytorch_model-00005-of-00005.safetensors"),
        ]
    ),
    ModelConfig(
        path=[
            os.path.join(MODEL_BASE, "Qwen-Image/text_encoder/model-00001-of-00004.safetensors"),
            os.path.join(MODEL_BASE, "Qwen-Image/text_encoder/model-00002-of-00004.safetensors"),
            os.path.join(MODEL_BASE, "Qwen-Image/text_encoder/model-00003-of-00004.safetensors"),
            os.path.join(MODEL_BASE, "Qwen-Image/text_encoder/model-00004-of-00004.safetensors"),
        ]
    ),
    ModelConfig(
        path=[
            os.path.join(MODEL_BASE, "Qwen-Image/vae/diffusion_pytorch_model.safetensors"),
        ]
    ),
]

PROCESSOR_CONFIG = ModelConfig(
    path=os.path.join(MODEL_BASE, "Qwen-Image-Edit/processor")
)
