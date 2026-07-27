import os
import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

# switch to "mps" for apple devices
# pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", dtype=torch.bfloat16, device="cuda:2", device_map="cuda")
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    torch_dtype=torch.bfloat16,
).to("cuda:2")

DATA_ROOT = "/path/to/local_storage/gpt5_data/qabench/"

# sampled_tasks = [
#     {   
#         "edit_path": os.path.join(DATA_ROOT, f"word_1_4/pair_01/prev.png"),
#         "prompt_path": os.path.join(DATA_ROOT, f"word_1_4/pair_01/qwen_prompt.txt"),
#         "output_path": f"example_res/image_lora_191_1202_word_1_4.jpg"
#     },
#     {
#         "edit_path": os.path.join(DATA_ROOT, "word_1_13/pair_02/prev.png"), 
#         "prompt_path": os.path.join(DATA_ROOT, "word_1_13/pair_02/qwen_prompt.txt"),
#         "output_path": "example_res/image_lora_191_1201_word_1_13.jpg"
#     },
#     {
#         "edit_path": os.path.join(DATA_ROOT, "word_1_67/pair_02/prev.png"), 
#         "prompt_path": os.path.join(DATA_ROOT, "word_1_67/pair_02/qwen_prompt.txt"),
#         "output_path": "example_res/image_lora_191_1201_word_1_67.jpg"
#     },
#     {
#         "edit_path": os.path.join(DATA_ROOT, "word_1_187/pair_01/prev.png"), 
#         "prompt_path": os.path.join(DATA_ROOT, "word_1_187/pair_01/qwen_prompt.txt"),
#         "output_path": "example_res/image_lora_191_1201_word_1_187.jpg"
#     },
# ]


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

sampled_tasks = [
     {
        "edit_path": os.path.join(DATA_ROOT, f"word_1_4/pair_0{i}/prev.png"),
        "prompt_path": os.path.join(DATA_ROOT, f"word_1_4/pair_0{i}/qwen_prompt.txt"),
        "output_path": f"example_res/n-1-base-diffuser/image_lora_191_1202_word_1_4_pait_0{i}.jpg"
    } for i in range(1, 8)
]

for t in tqdm(sampled_tasks):
    with open(t["prompt_path"], "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    input_image = load_image(t["edit_path"])
    image = pipe(image=input_image, prompt=prompt).images[0]
    os.makedirs(os.path.dirname(t["output_path"]), exist_ok=True)
    image.save(t["output_path"])
    print(f"Saved: {t['output_path']}")

