import os
from tqdm import tqdm
from utils.call_image_gen_api import generate_image
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_folders(root_path):
    folders = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if 'prev.png' in filenames:
            folders.append(dirpath)
    return folders

def generate_and_save(folder):
    prev_path = os.path.join(folder, "prev.png")
    prompt_path = os.path.join(folder, "prompt_nl_gt.txt")
    output_path = os.path.join(folder, "generated_next/next_gpt-image-1.5_prompt_nl_gt.png")
    
    if os.path.exists(output_path):
        print(f"Image already exists at {output_path}, skipping generation.")
        return
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    
    try:
        img = generate_image(
            image_paths=[prev_path],
            prompt=prompt,
            size=None,
            model_name='gpt-image-1.5',
            n_gen=1
        )
        img.save(output_path)
        print(f"Generated image saved to {output_path}")
    except Exception as e:
        print(f"Failed to generate image for folder {folder}: {e}")

if __name__ == "__main__":
    root_path = os.environ.get("DATA_ROOT", "data") + "/new_prompt/test_mini"
    folders = find_folders(root_path)
    print(f"Found {len(folders)} folders to process")
    
    max_workers = 5  # Adjust based on your API rate limits
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_and_save, folder): folder for folder in folders}
        
        for future in tqdm(as_completed(futures), total=len(folders)):
            folder = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {folder}: {e}")