import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import os
from tqdm import tqdm

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=None,
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)
epoch = 6
lora_root = os.environ.get("LORA_ROOT", "/path/to/lora_checkpoints")
pipe.load_lora(pipe.dit, os.path.join(lora_root, f"epoch-{epoch}.safetensors"))

DATA_ROOT = "/path/to/local_storage/gpt5_data/qabench/"

sampled_tasks = [
    {   
        "edit_path": os.path.join(DATA_ROOT, f"word_1_4/pair_01/prev.png"),
        "prompt_path": os.path.join(DATA_ROOT, f"word_1_4/pair_01/qwen_prompt.txt"),
        "output_path": f"example_res/image_lora_191_1202_epoch{epoch}_word_1_4.jpg"
    },
    {
        "edit_path": os.path.join(DATA_ROOT, "word_1_13/pair_02/prev.png"), 
        "prompt_path": os.path.join(DATA_ROOT, "word_1_13/pair_02/qwen_prompt.txt"),
        "output_path": f"example_res/image_lora_191_1202_epoch{epoch}_word_1_13.jpg"
    },
    {
        "edit_path": os.path.join(DATA_ROOT, "word_1_67/pair_02/prev.png"), 
        "prompt_path": os.path.join(DATA_ROOT, "word_1_67/pair_02/qwen_prompt.txt"),
        "output_path": f"example_res/image_lora_191_1202_epoch{epoch}_word_1_67.jpg"
    },
    {
        "edit_path": os.path.join(DATA_ROOT, "word_1_187/pair_01/prev.png"), 
        "prompt_path": os.path.join(DATA_ROOT, "word_1_187/pair_01/qwen_prompt.txt"),
        "output_path": f"example_res/image_lora_191_1202_epoch{epoch}_word_1_187.jpg"
    },
]


# sampled_tasks = [
#     {
#         "edit_path": os.path.join(DATA_ROOT, f"word_1_4/pair_0{i}/prev.png"),
#         "prompt_path": os.path.join(DATA_ROOT, f"word_1_4/pair_0{i}/qwen_prompt.txt"),
#         "output_path": f"example_res/n-1/image_lora_191_1202_word_1_4_pait_0{i}.jpg"
#     } for i in range(1, 8)
# ]

# sampled_tasks = [
#     {
#         "edit_path": os.path.join(DATA_ROOT, f"word_1_4/pair_0{i}/prev.png") if i == 1 else f"example_res/n-k-base/image_lora_191_1202_word_1_4_pait_0{i-1}.jpg",
#         "prompt_path": os.path.join(DATA_ROOT, f"word_1_4/pair_0{i}/qwen_prompt.txt"),
#         "output_path": f"example_res/n-k-base1/image_lora_191_1202_word_1_4_pait_0{i}.jpg"
#     } for i in range(1, 8)
# ]

# sampled_tasks = [
#      {
#         "edit_path": os.path.join(DATA_ROOT, f"word_1_4/pair_0{i}/prev.png"),
#         "prompt_path": os.path.join(DATA_ROOT, f"word_1_4/pair_0{i}/qwen_prompt.txt"),
#         "output_path": f"example_res/n-1-test/image_lora_191_1202_word_1_4_pait_0{i}.jpg"
#     } for i in range(1, 8)
# ]

for t in tqdm(sampled_tasks):
    with open(t["prompt_path"], "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    edit_image = Image.open(t["edit_path"]).convert("RGB")
    print(f"prompt: {prompt}")
    image = pipe(prompt, edit_image=edit_image, seed=123, num_inference_steps=40, height=edit_image.size[1], width=edit_image.size[0])
    # image = pipe(prompt, edit_image=edit_image, seed=123, num_inference_steps=40, edit_image_auto_resize=True)
    os.makedirs(os.path.dirname(t["output_path"]), exist_ok=True)
    image.save(t["output_path"])
    print(f"Saved: {t['output_path']}")
