import json
import os
from utils.cloudgpt_aoai import get_chat_completion, encode_image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts.textual_wm import TEXT_GT_GENERATION_PROMPT
from prompts.gui_descriptions import GUI_DESCRIPTION

def find_folders(root_path):
    folders = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if 'prev.png' in filenames:
            folders.append(dirpath)
    return folders

def generate_nl_description(
    folder, 
    llm_model: str = "gpt-5.2-chat-20251211",
    output_name: str = "prompt_nl_gt.txt"
):
    if os.path.exists(os.path.join(folder, output_name)):
        print(f"Skipping existing: {folder}")
        return
    
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
    
    with open(os.path.join(folder, 'action.json'), 'r') as f:
        action = json.load(f)
    
    function = action['function']
    
    if function not in GUI_DESCRIPTION[app]:
        gui_description = "No description available."
    else:
        gui_description = GUI_DESCRIPTION[app][function]
    
    prompt = TEXT_GT_GENERATION_PROMPT.format(
        app_name=app_name,
        action=json.dumps(action),
        gui_description=gui_description
    )
    
    prev_url = encode_image(os.path.join(folder, 'prev.png'))
    next_url = encode_image(os.path.join(folder, 'next.png'))
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": prev_url}},
                {"type": "image_url", "image_url": {"url": next_url}},
            ],
        }
    ]
    
    response = get_chat_completion(
        model=llm_model,
        messages=messages,
        use_broker_login=False
    )
    
    content = response.choices[0].message.content.strip()
    
    with open(os.path.join(folder, output_name), 'w') as f:
        f.write(content)
    
    return content, prompt

if __name__ == "__main__":
    root = "/path/to/cuwm/data_sample_3000/new_prompt/test"
    folders = find_folders(root)
    
    max_workers = 15 # 调整并行数量
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_nl_description, folder): folder for folder in folders}
        
        for future in tqdm(as_completed(futures), total=len(folders)):
            result = future.result()
            # folder = futures[future]
            # try:
            #     result = future.result()
            #     print(f"Completed: {folder}")
            # #     assert 0
            # except Exception as e:
            #     print(f"Error processing {folder}: {e}")