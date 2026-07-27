from utils.cloudgpt_aoai import get_openai_client
import base64
from pathlib import Path
import io
from PIL import Image

def generate_image(
    image_paths: list[str],
    prompt: str, 
    size: str, 
    model_name: str,
    n_gen: int = 1,
) -> bool:
    assert model_name in ["dall-e-3", "gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5"], f"Unsupported model: {model_name}"
    try:
        client = get_openai_client()
        
        if model_name == "dall-e-3":
            raise NotImplementedError("DALL-E-3 image generation not implemented yet.")
            # DALL-E-3 only supports single image with base64 response
            results = client.images.generate(
                model=model_name,
                prompt=prompt,
                size=size,
                response_format="b64_json",
            )
        else:
            results = client.images.edit(
                model=model_name,
                image=[Path(img) for img in image_paths],
                prompt=prompt,
                n=n_gen,
                size='auto',
            )
        
        b64 = results.data[0].b64_json
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
        return img
        
    except Exception as e:
        print(f"Error generating image: {e}")
        raise e
