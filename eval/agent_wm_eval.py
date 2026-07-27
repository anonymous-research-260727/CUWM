import argparse
import json
import logging
import math
import os

from tqdm import tqdm

from eval.action_evaluation import ActionEvaluation, ActionEvaluationA11y

logger = logging.getLogger(__name__)

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

def get_gt_action(folder, eval_dir, source_data_root=None):
    if source_data_root is None:
        source_data_root = os.environ.get("SOURCE_DATA_ROOT", "data")
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
    
    with open(os.path.join(folder, "action_options_raw.json"), "r") as f:
        data = json.load(f)
        
    if 0 <= action_idx < len(data):        
        action_option = data[action_idx]["tool_call"]
    else:
        logger.warning("Invalid action idx %d for folder %s with file %s.", action_idx, folder, key)
        action_option = {"function": "invalid_function", "args": {}, "status": "invalid_status"}
        action_idx = -1
        
    return action_option, action_idx
    
def get_all_action_options(folder):
    with open(os.path.join(folder, "action_options_raw.json"), "r") as f:
        action_options = json.load(f)
    return [option["tool_call"] for option in action_options]

def evaluate_actions(filenames, folders, eval_dir, evaluator):
    overall_result = {}
    skipped_folders = set()
    
    # Load the best/worse results file once
    best_worse_path = os.path.join(eval_dir, "eval_result/agent_wm_gt_action.json")
    best_worse_data = {}
    if os.path.isfile(best_worse_path):
        with open(best_worse_path, "r") as f:
            best_worse_data = json.load(f)
            
    ref_key = "action_selection_none.json"

    count_missing_gt = 0

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
            action_command_res["2_match"] = action_command_res["function_match"] and action_command_res["args_match"]
            action_command_res["pred_action_idx"] = pred_idx
            
            # Check gt_idx validation using pre-loaded data
            if ref_key in best_worse_data and folder in best_worse_data[ref_key]:
                gt_indices = best_worse_data[ref_key][folder].get("gt_idx", [])
                action_command_res["gt_idx_match"] = 1.0 if pred_idx in gt_indices else 0.0
            else:
                action_command_res["gt_idx_match"] = 0.0 # Default if not found
                # Optional: debug print if missing too often
                # count_missing_gt += 1

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent WM Evaluation")
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Path to the evaluation directory"
    )
    parser.add_argument(
        "--source_data_root",
        type=str,
        default=None,
        help="Path to source data root (defaults to DATA_ROOT env var or 'data')"
    )
    args = parser.parse_args()

    source_data_root = args.source_data_root or os.environ.get("SOURCE_DATA_ROOT", "data")
    eval_dir = args.eval_dir
    
    assert os.path.isdir(eval_dir), f"Eval dir {eval_dir} does not exist."
    
    
    filenames = [
        # "action_selection_none.json",
        # "action_selection_text.json",
        "generated_base/action_selection_image.json",
        # "generated_base/action_selection_text+image.json",
        # "generated_epoch-24/action_selection_image.json",
        # "generated_epoch-24/action_selection_text+image.json",
        # "generated_gpt-image-1.5/action_selection_image.json",
        # "generated_gpt-image-1.5/action_selection_text+image.json",
    ]
    
    
    folders = find_folders(eval_dir, filenames)
    logger.info("%d folders found for evaluation.", len(folders))
    # assert 0
    
    evaluator = ActionEvaluationA11y()
    evaluate_actions(filenames, folders, eval_dir, evaluator)
    get_best_worse_score([filenames[0]], folders, eval_dir, evaluator)
    

