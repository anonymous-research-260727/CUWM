import os
import json
from tqdm import tqdm
import math
from eval.action_evaluation import ActionEvaluation, ActionEvaluationA11y

def find_folders(root_path, filekeys):
    folders = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        # print(filenames, filekeys)
        flag = "action_options.json" in filenames
        for filename in filekeys:
            if not os.path.isfile(os.path.join(dirpath, filename)):
                flag = False
                break
        if flag:
            folders.append(dirpath)
    return folders

def get_gt_action(folder, eval_dir):
    rel_path = os.path.relpath(folder, eval_dir)
    action_file = os.path.join(source_data_root, rel_path, "action.json")
    action_raw = os.path.join(source_data_root, rel_path, "action_raw.json")
    
    with open(action_file, "r") as f:
        action_command = json.load(f)
    with open(action_raw, "r") as f:
        action_raw_data = json.load(f)
    return action_command, action_raw_data

def get_pred_action(folder, key):
    action_idx_file = os.path.join(folder, key)
    # print(action_idx_file)
    if not os.path.isfile(action_idx_file):
        return None
    
    with open(action_idx_file, "r") as f:
        action_data = json.load(f)
    action_idx = action_data["action_idx"]
    
    with open(os.path.join(folder, "action_options.json"), "r") as f:
        data = json.load(f)
        
    if 0 <= action_idx < len(data):        
        action_option = data[action_idx]
    else:
        print(f"Invalid action idx {action_idx} for folder {folder} with file {key}.")
        action_option = {"function": "invalid_function", "args": {}, "status": "invalid_status"}
        action_idx = -1
        
    return action_option, action_idx
    
def get_all_action_options(folder):
    with open(os.path.join(folder, "action_options.json"), "r") as f:
        action_options = json.load(f)
    return action_options

def evaluate_actions(filenames, folders, eval_dir, evaluator):
    overall_result = {}
    skipped_folders = set()
    
    for filename in filenames:
        overall_result[filename] = {}
        
        for folder in tqdm(folders):
            # print(f"Evaluating {folder} with {filename}...")
            gt_command, gt_raw = get_gt_action(folder, eval_dir)
            pred_command, pred_idx = get_pred_action(folder, filename)
            
            action_command_res = evaluator.compare_action_command(
                gt_raw=gt_raw,
                gt_command=gt_command,
                pred_command=pred_command,
            )
            action_command_res["pred_action_idx"] = pred_idx
            
            overall_result[filename][folder] = action_command_res
    
    # 保存结果
    os.makedirs(os.path.join(eval_dir, "eval_result"), exist_ok=True)
    
    with open(os.path.join(eval_dir, "eval_result/agent_wm_evaluation_results.json"), "w") as f:
        json.dump(overall_result, f, indent=2)
    
    # summary the results
    summary_result = {}
    for filename, folder_results in overall_result.items():
        summary_result[filename] = {}
        count = len(folder_results)
        for folder, metrics in folder_results.items():
            if folder in skipped_folders:
                continue
            for key, value in metrics.items():
                if key != "pred_action_idx":
                    if key not in summary_result[filename]:
                        summary_result[filename][key] = 0.0
                    summary_result[filename][key] += value / count
    
    with open(os.path.join(eval_dir, "eval_result/agent_wm_evaluation_summary.json"), "w") as f:
        json.dump(summary_result, f, indent=2)
    
    return overall_result, summary_result

def get_best_worse_score(filename, folders, eval_dir, evaluator):
    overall_result = {}
    skipped_folders = set()
    
    for filename in filenames:
        overall_result[filename] = {}
        
        for folder in tqdm(folders):
            # print(f"Evaluating {folder} with {filename}...")
            gt_command, gt_raw = get_gt_action(folder, eval_dir)
            action_options = get_all_action_options(folder)
            
            best_score, worst_score = -math.inf, math.inf
            
            result = {"gt_idx": []}
            for idx, option in enumerate(action_options):
                
                pred_command = option
                
                action_command_res = evaluator.compare_action_command(
                    gt_raw=gt_raw,
                    gt_command=gt_command,
                    pred_command=pred_command,
                )["overall_match"]
                
                action_command_res = int(action_command_res)
                
                if action_command_res == 1:
                    result["gt_idx"].append(idx)
            
                if action_command_res > best_score:
                    best_score = action_command_res
                if action_command_res < worst_score:
                    worst_score = action_command_res
            
            result["best_score"] = best_score
            result["worst_score"] = worst_score
            overall_result[filename][folder] = result
    
    # 保存结果
    with open(os.path.join(eval_dir, "eval_result/agent_wm_best_worse_results.json"), "w") as f:
        json.dump(overall_result, f, indent=2)
    
    # summary the results
    summary_result = {}
    
    for filename, folder_results in overall_result.items():
        summary_result[filename] = {}
        summary_result[filename]["gt_idx"] = {}
        gt_count = 0
        count = len(folder_results)
        for folder, metrics in folder_results.items():
            if folder in skipped_folders:
                continue
            for key, value in metrics.items():
                if key != "gt_idx":
                    if key not in summary_result[filename]:
                        summary_result[filename][key] = 0.0
                    summary_result[filename][key] += value / count
                if key == "gt_idx":
                    if len(value) > 0:
                        gt_count += 1
                    for gt_idx in value:
                        if gt_idx not in summary_result[filename]["gt_idx"]:
                            summary_result[filename]["gt_idx"][gt_idx] = 0
                        summary_result[filename]["gt_idx"][gt_idx] += 1
    
        summary_result[filename]["gt_idx_ratio"] = {}
        for gt_idx, cnt in summary_result[filename]["gt_idx"].items():
            ratio = cnt / gt_count if gt_count > 0 else 0
            summary_result[filename]["gt_idx_ratio"][gt_idx] = ratio
                    
    
    with open(os.path.join(eval_dir, "eval_result/agent_wm_best_worse_summary.json"), "w") as f:
        json.dump(summary_result, f, indent=2)
    
    return overall_result, summary_result


def load_banana_completed_folders(banana_completed_file):
    """Load the list of folders completed by Nano Banana."""
    completed_folders = []
    with open(banana_completed_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                completed_folders.append(line)
    return completed_folders


def filter_folders_by_banana_completed(folders, eval_dir, banana_completed_folders):
    """Filter folders to only include those completed by Nano Banana."""
    filtered = []

    for folder in folders:
        # Get relative path from eval_dir
        rel_path = os.path.relpath(folder, eval_dir)

        # Check if this relative path is in the banana completed list
        if rel_path in banana_completed_folders:
            filtered.append(folder)

    return filtered


if __name__ == "__main__":
    # Update these paths for your environment
    import sys

    # Portable defaults; override with the positional arguments below.
    source_data_root = os.environ.get("SOURCE_DATA_ROOT", "data/test_mini")
    eval_dir = os.environ.get("EVAL_DIR", "outputs/test_mini")
    banana_completed_file = os.environ.get("COMPLETED_FILE", "outputs/completed.txt")

    # Allow command-line override
    if len(sys.argv) > 1:
        eval_dir = sys.argv[1]
    if len(sys.argv) > 2:
        source_data_root = sys.argv[2]
    if len(sys.argv) > 3:
        banana_completed_file = sys.argv[3]

    assert os.path.isdir(eval_dir), f"Eval dir {eval_dir} does not exist."

    # Load Nano Banana completed folders for fair comparison
    print(f"Loading Nano Banana completed folders from: {banana_completed_file}")
    banana_completed_folders = load_banana_completed_folders(banana_completed_file)
    print(f"Loaded {len(banana_completed_folders)} completed Nano Banana folders\n")

    # Update filenames based on what you actually generated
    filenames = [
        "action_selection_none.json",
        "action_selection_text.json",
        # "generated_base/action_selection_image.json",
        # "generated_base/action_selection_text+image.json",
        # "generated_epoch-24/action_selection_image.json",
        # "generated_epoch-24/action_selection_text+image.json",
    ]

    # Find all folders with the required files
    all_folders = find_folders(eval_dir, filenames)
    print(f"Found {len(all_folders)} folders with required files in {eval_dir}")

    # Filter to only Nano Banana completed folders
    folders = filter_folders_by_banana_completed(all_folders, eval_dir, banana_completed_folders)
    print(f"Filtered to {len(folders)} folders completed by Nano Banana (fair comparison set)\n")

    if len(folders) == 0:
        print("WARNING: No matching folders found! Check that:")
        print(f"  1. {banana_completed_file} exists and has content")
        print(f"  2. Paths in banana_completed.txt match the structure in {eval_dir}")
        exit(1)

    evaluator = ActionEvaluationA11y()
    print("Starting evaluation...")
    evaluate_actions(filenames, folders, eval_dir, evaluator)
    print("\nCalculating best/worst scores...")
    get_best_worse_score([filenames[0]], folders, eval_dir, evaluator)

    print(f"\n✓ Evaluation complete! Results saved to: {eval_dir}/eval_result/")
    print(f"  - agent_wm_evaluation_summary.json (main metrics)")
    print(f"  - agent_wm_best_worse_summary.json (oracle performance)")
    

