from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import torch
import os
from tqdm import tqdm

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

# image_1 = pipe(prompt="一位少女", seed=0, num_inference_steps=40, height=1328, width=1024)
# image_1.save("image1.jpg")

# image_2 = pipe(prompt="一位老人", seed=0, num_inference_steps=40, height=1328, width=1024)
# image_2.save("image2.jpg")

# prompt = "生成这两个人的合影"
# edit_image = [Image.open("image1.jpg"), Image.open("image2.jpg")]
# image_3 = pipe(prompt, edit_image=edit_image, seed=1, num_inference_steps=40, height=1328, width=1024, edit_image_auto_resize=True)
# image_3.save("image3.jpg")

DATA_ROOT = "/path/to/local_storage/gpt5_data/qabench/"

# sampled_tasks = [
#     {
#         "edit_path": os.path.join(DATA_ROOT, f"word_1_4/pair_0{i}/prev.png") if i == 1 else f"example_res/n-k-base1/image_lora_191_1202_word_1_4_pait_0{i-1}.jpg",
#         "prompt_path": os.path.join(DATA_ROOT, f"word_1_4/pair_0{i}/qwen_prompt.txt"),
#         "output_path": f"example_res/n-k-base1/image_lora_191_1202_word_1_4_pait_0{i}.jpg"
#     } for i in range(1, 8)
# ]

sampled_tasks = [
     {
        "edit_path": os.path.join(DATA_ROOT, f"word_1_4/pair_0{i}/prev.png"),
        "prompt_path": os.path.join(DATA_ROOT, f"word_1_4/pair_0{i}/qwen_prompt.txt"),
        "output_path": f"example_res/n-1-base-test/image_lora_191_1202_word_1_4_pait_0{i}.jpg"
    } for i in range(1, 8)
]

for t in tqdm(sampled_tasks):
    with open(t["prompt_path"], "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    edit_image = Image.open(t["edit_path"]).convert("RGB")
    image = pipe(prompt, edit_image=edit_image, seed=123, num_inference_steps=40, height=edit_image.size[1], width=edit_image.size[0])
    os.makedirs(os.path.dirname(t["output_path"]), exist_ok=True)
    image.save(t["output_path"])
    print(f"Saved: {t['output_path']}")