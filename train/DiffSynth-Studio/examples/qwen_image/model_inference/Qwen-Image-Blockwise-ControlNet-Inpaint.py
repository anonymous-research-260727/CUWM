import torch
from PIL import Image
from modelscope import dataset_snapshot_download
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, ControlNetInput


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint", origin_file_pattern="model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_image_dataset",
    local_dir="./data/example_image_dataset",
    allow_file_pattern="inpaint/*.jpg"
)
# prompt = "a cat with sunglasses"
# controlnet_image = Image.open("./data/example_image_dataset/inpaint/image_1.jpg").convert("RGB").resize((1328, 1328))
# inpaint_mask = Image.open("./data/example_image_dataset/inpaint/mask.jpg").convert("RGB").resize((1328, 1328))

controlnet_image = Image.open("/path/to/local_storage/gpt5_data/qabench/word_1_4/pair_01/prev.png").convert("RGB").resize((1024, 736))
# inpaint_mask = Image.open("/path/to/local_storage/bbox_mask.png").convert("RGB").resize((1024, 736))
inpaint_mask = Image.open("/path/to/local_storage/mask.png").convert("RGB").resize((1024, 736))
prompt = "Generate the next frame of this Microsoft Word window based on the previous screenshot. In the new frame, the user has clicked the “Customize Quick Access Toolbar” button located near the upper-right corner of the Word interface, just to the left of the “Share” button. This action opens a vertical dropdown menu.  Maintain the same overall layout, colors, and styling as the input image — the Word window remains maximized, with the “Home” tab selected, the same toolbar icons visible, and the same document contents, highlighting, and comments preserved exactly as shown. The text in the document should remain unchanged, keeping identical cursor position, selection highlight, text formatting, and comments panel on the right side. Add the new element: a right-aligned dropdown menu labeled “Customize Quick Access Toolbar.” It should appear directly below the small down-arrow button that triggers it. The menu should have a clean white background, black text, and show a list of options vertically. Some items have checkmarks next to them, indicating they are currently active. The visible menu options should include entries such as “Automatically Save,” “New,” “Open,” “Save,” “Email,” “Quick Print,” “Print Preview and Print,” “Editor (F7),” “Read Aloud,” “Undo,” “Redo,” “Draw Table,” “Touch/Mouse Mode,” “More Commands…,” “Show above the ribbon,” “Hide Quick Access Toolbar,” and “Use Removed Tools.” Ensure checkmarks appear next to “Automatically Save,” “Save,” and “Read Aloud.” The dropdown extends downward from the top toolbar area but does not obscure the document text. Lighting, shadows, and window borders must look natural, consistent with the previous frame. The dropdown should stand out clearly with sharp edges, appropriate spacing between list items, and smooth contrast against the background. Preserve all other interface details exactly — the user profile button “U” remains visible, comments on the right side stay in place, and document zoom level and bottom status bar remain unchanged. Overall, the generated image must depict the result immediately after clicking the “Customize Quick Access Toolbar” button, accurately showing the opened customization menu while keeping every other visual and contextual element identical to the previous frame.",
# image = pipe(prompt, edit_image=edit_image, seed=123, num_inference_steps=40, height=edit_image.size[1], width=edit_image.size[0])
# image.save("image_inpaint.jpg")

image = pipe(
    prompt, seed=0,
    input_image=controlnet_image, inpaint_mask=inpaint_mask,
    blockwise_controlnet_inputs=[ControlNetInput(image=controlnet_image, inpaint_mask=inpaint_mask)],
    num_inference_steps=40,
    height=controlnet_image.size[1], width=controlnet_image.size[0]
)
image.save("image_word_1_4_inpaint_1.jpg")
