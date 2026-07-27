# server_single.py
import os
import time
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

# =============================
# 配置（与你离线脚本一致）
# =============================
DTYPE = torch.bfloat16

from model_configs import MODEL_CONFIGS, PROCESSOR_CONFIG

LORA_EPOCH = 24
LORA_PATH = os.path.join(
    os.environ.get("LORA_DIR", "models/lora"),
    f"epoch-{LORA_EPOCH}.safetensors"
)

DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/qwen_image_edit_api")

# =============================
# FastAPI
# =============================
app = FastAPI()
pipe: Optional[QwenImagePipeline] = None


class EditReq(BaseModel):
    image_path: str
    prompt_path: str
    output_path: Optional[str] = None


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


@app.on_event("startup")
def startup():
    global pipe

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    # 单实例：用当前 CUDA_VISIBLE_DEVICES 下的第 0 张
    torch.cuda.set_device(0)

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=DTYPE,
        device="cuda:0",
        model_configs=MODEL_CONFIGS,
        tokenizer_config=None,
        processor_config=PROCESSOR_CONFIG,
    )

    pipe.load_lora(pipe.dit, LORA_PATH)

    pipe.dit.eval()
    torch.set_grad_enabled(False)


@app.post("/v1/edit_by_path")
def edit_by_path(req: EditReq):
    assert pipe is not None, "pipeline not initialized"

    if not os.path.isfile(req.image_path):
        return {"ok": False, "error": f"image not found: {req.image_path}"}

    if not os.path.isfile(req.prompt_path):
        return {"ok": False, "error": f"prompt not found: {req.prompt_path}"}

    image = Image.open(req.image_path).convert("RGB")
    prompt = read_text(req.prompt_path)

    if req.output_path:
        out_path = req.output_path
    else:
        base = os.path.splitext(os.path.basename(req.image_path))[0]
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{base}_{ts}.png")

    ensure_dir(out_path)

    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
            # ✅ 完全复用你原来的调用方式
            out = pipe(
                prompt,
                edit_image=image,
                seed=123,
                num_inference_steps=40,
                height=image.size[1],
                width=image.size[0],
            )

        out_img = out.images[0] if hasattr(out, "images") else out[0]
        out_img.save(out_path)

        return {
            "ok": True,
            "output_path": out_path,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }
