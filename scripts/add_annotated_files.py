import os
import shutil

def find_folders(root_path):
    folders = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if 'prev.png' in filenames:
            folders.append(dirpath)
    return folders


def get_annotated_file_path(folder):
    parts = folder.split('/')
    app = parts[-5]
    domain = parts[-4]
    task = parts[-2]
    pair = parts[-1]
    pair_id = int(pair.split('_')[-1])
    folder = f"{SOURCE_ROOT}/image/{app}/{domain}/success/{task}"
    annotated_files = sorted([f for f in os.listdir(folder) if f.endswith('_annotated.png')])
    return {
        "prev_annotated": os.path.join(folder, annotated_files[pair_id - 1]),
        "next_annotated": os.path.join(folder, annotated_files[pair_id]),
    }
  

if __name__ == "__main__":
    SOURCE_ROOT = os.environ.get("SOURCE_ROOT", "data/test")
    TARGET_ROOT = os.environ.get("TARGET_ROOT", "data/new_prompt/test")
    folders = find_folders(TARGET_ROOT)
    for folder in folders:
        paths = get_annotated_file_path(folder)
        prev_annoatated_source_path = paths["prev_annotated"]
        next_annotated_source_path = paths["next_annotated"]
        
        prev_annoatated_target_path = os.path.join(folder, "prev_annotated.png")
        next_annotated_target_path = os.path.join(folder, "next_annotated.png")
        
        assert not os.path.exists(prev_annoatated_target_path)
        assert not os.path.exists(next_annotated_target_path)
        
        shutil.copy2(prev_annoatated_source_path, prev_annoatated_target_path)
        shutil.copy2(next_annotated_source_path, next_annotated_target_path)
