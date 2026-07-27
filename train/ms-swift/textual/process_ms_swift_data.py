import os
import json
import glob
from prompts.textual_wm import TEXT_PRED_PREDICTION_PROMPT
from prompts.gui_descriptions import GUI_DESCRIPTION
from tqdm import tqdm

def find_folders(root_path):
    folders = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if 'prev.png' in filenames:
            folders.append(dirpath)
    return folders

def build_dataset(data_root: str):
    """递归遍历所有 pair_* 文件夹，构建数据"""
    samples = []

    pair_dirs = find_folders(data_root)

    for idx, pair_dir in tqdm(enumerate(pair_dirs)):
        prev_path = os.path.join(pair_dir, "prev.png")
        next_path = os.path.join(pair_dir, "next.png")
        action_path = os.path.join(pair_dir, "action.json")
        response_path = os.path.join(pair_dir, "prompt_nl_gt.txt")
        
        assert os.path.exists(prev_path), f"Missing {prev_path}"
        assert os.path.exists(next_path), f"Missing {next_path}"
        assert os.path.exists(action_path), f"Missing {action_path}"
        assert os.path.exists(response_path), f"Missing {response_path}"
        
        with open(action_path, "r", encoding="utf-8") as f:
            action = json.load(f)
            
        if "word" in pair_dir.lower():
            app = "word"
            app_name = "Microsoft Word"
        elif "excel" in pair_dir.lower():
            app = "excel"
            app_name = "Microsoft Excel"
        elif "ppt" in pair_dir.lower():
            app = "ppt"
            app_name = "Microsoft PowerPoint"
        else:
            raise ValueError(f"Unknown application type in folder: {pair_dir}")
        
        gui_description = GUI_DESCRIPTION[app][action["function"]]

        prompt = TEXT_PRED_PREDICTION_PROMPT.format(
            app_name=app_name,
            image="<image>",
            action=json.dumps(action),
            gui_description=gui_description
        )
        
        with open(response_path, "r", encoding="utf-8") as f:
            response = f.read().strip()
            
        
        sample = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ],
            "images": [
                prev_path
            ]
        }

        samples.append(sample)

    return samples


if __name__ == "__main__":
    # ===== 根路径 =====
    DATA_ROOT = "/path/to/cuwm/data_sample_3000/new_prompt/train"
    os.makedirs("data", exist_ok=True)
    
    train_data = build_dataset(DATA_ROOT)
    print(f"✅ Training samples: {len(train_data)} ")
    with open(f"data/train_data_1231_{len(train_data)}.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print("✅ Training data saved.")
    
    # ===== 根路径 =====
    DATA_ROOT = "/path/to/cuwm/data_sample_3000/new_prompt/test"
    
    test_data = build_dataset(DATA_ROOT)
    print(f"✅ Testing samples: {len(test_data)} ")
    with open(f"data/test_data_1231_{len(test_data)}.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print("✅ Testing data saved.")