import json
import logging
import os
from utils.cloudgpt_aoai import get_chat_completion, encode_image
from tqdm import tqdm
from prompts.gui_descriptions import GUI_DESCRIPTION
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts.textual_wm import TEXT_PRED_PREDICTION_PROMPT

logger = logging.getLogger(__name__)


def find_folders(root_path):
    folders = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if 'prev.png' in filenames:
            folders.append(dirpath)
    return folders

def generate_nl_description(
    folder,
    action=None,
    client=None,
    llm_model: str = "gpt-5.2-chat-20251211",
    temperature: float = 0,
    lora_path: str = None,
):
    from utils.qwen_local_api import get_qwen_chat_completion, is_qwen_model

    if "word" in folder.lower():
        app = "word"
        app_name = "Microsoft Word"
    elif "excel" in folder.lower():
        app = "excel"
        app_name = "Microsoft Excel"
    elif "ppt" in folder.lower():
        app = "ppt"
        app_name = "Microsoft PowerPoint"
    else:
        raise ValueError(f"Unknown application type in folder: {folder}")

    if action is None:
        with open(os.path.join(folder, 'action.json'), 'r') as f:
            action = json.load(f)
    function = action['function']

    if function not in GUI_DESCRIPTION[app]:
        gui_description = "No description available."
    else:
        gui_description = GUI_DESCRIPTION[app][function]

    prompt = TEXT_PRED_PREDICTION_PROMPT.format(
        image="See the attached image.",
        app_name=app_name,
        action=json.dumps(action),
        gui_description=gui_description
    )

    # Check if using Qwen VL model or local path
    use_qwen = is_qwen_model(llm_model) or llm_model.startswith("/")

    if "qwen2.5-vl-7b-instruct" in llm_model.lower():
        from utils.qwen_dashscope_7b_api import get_qwen_dashscope_7b_response as get_qwen_7b_response
        logger.debug("Generating text synthesis using Qwen 7B API: %s", llm_model)
        # Use OpenAI format - base64 encoded images (DashScope supports this)
        image_url = encode_image(os.path.join(folder, 'prev.png'))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        response = get_qwen_7b_response(
            messages=messages,
            model=llm_model,
            temperature=temperature,
        )
    elif use_qwen:
        logger.debug("Generating text synthesis using Qwen model: %s", llm_model)
        # Use Qwen format - image paths directly
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": os.path.join(folder, 'prev.png')},
                ],
            }
        ]
        response = get_qwen_chat_completion(
            messages=messages,
            model=llm_model,
            temperature=temperature,  # Use temperature 0 for deterministic output
            lora_path=lora_path,
        )
    else:
        logger.debug("Generating text synthesis using OpenAI model: %s", llm_model)
        # Use OpenAI format - base64 encoded images
        image_url = encode_image(os.path.join(folder, 'prev.png'))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        response = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=temperature,  # Use temperature 0 for deterministic output
        )
        

    content = response.choices[0].message.content.strip()
    return content, prompt

def generate_nl_description_prompt(folder, action: dict = None):
    raise NotImplementedError("改过了generate_nl_description_prompt，需要同步修改这个函数")
    with open(os.path.join(folder, 'action.json'), 'r') as f:
        action = json.load(f)  
    
    control_info = None
    if os.path.exists(os.path.join(folder, 'control_info.json')):
        with open(os.path.join(folder, 'control_info.json'), 'r') as f:
            control_info = json.load(f)

    content, prompt = generate_nl_description(
        folder,
        action=action,
        control_info=control_info
    )
    
    with open(os.path.join(folder, 'prompt_nl_pred.txt'), 'w') as f:
        f.write(content)
        
    return content

if __name__ == "__main__":
    folders = find_folders(os.environ.get("DATA_ROOT", "data") + "/new_prompt/test_mini")
    
    max_workers = 10  # 调整并行数量
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_nl_description_prompt, folder): folder for folder in folders}
        
        for future in tqdm(as_completed(futures), total=len(folders)):
            folder = futures[future]
            try:
                result = future.result()
                logger.info("Completed: %s", folder)
            except Exception as e:
                logger.error("Error processing %s: %s", folder, e)