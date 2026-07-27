import json
import os
from pathlib import Path

# Constants
_data_root = os.environ.get("DATA_ROOT", "data")
NEW_EXP_BASE = os.environ.get("NEW_EXP_BASE", os.path.join(_data_root, "agent_eval_result/new_exp/new_prompt/test"))
OLD_EXP_BASE = os.environ.get("OLD_EXP_BASE", os.path.join(_data_root, "agent_eval_result/old_exp/new_prompt/test"))
GT_FILE = os.environ.get("GT_FILE", os.path.join(OLD_EXP_BASE, "eval_result/agent_wm_best_worse_results.json"))

def normalize_action(action):
    """
    Normalize an action to a comparable format.
    Converts args dict to sorted JSON string for comparison.
    """
    normalized = {
        "function": action.get("function", ""),
        "args": json.dumps(action.get("args", {}), sort_keys=True),
        # Ignore status for comparison as it might not be relevant for uniqueness/correctness
    }
    return json.dumps(normalized, sort_keys=True)

def main():
    # 1. Load GT file
    print(f"Loading GT file from {GT_FILE}")
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    # The GT data is keyed by pair directory within "action_selection_none.json" (or others, but we assume none for now based on user context)
    # Actually user didn't specify mode, but let's assume "action_selection_none.json" based on the head command output
    gt_map = gt_data.get("action_selection_none.json", {})
    
    # 2. Find all pairs in NEW_EXP_BASE
    print(f"Scanning pairs in {NEW_EXP_BASE}")
    new_pairs = []
    base_path = Path(NEW_EXP_BASE)
    for action_options_path in base_path.rglob("action_options.json"):
        new_pairs.append(action_options_path.parent)

    print(f"Found {len(new_pairs)} pairs to check.")
    
    found_count = 0
    missing_count = 0
    error_count = 0
    
    print("\n" + "="*80)
    print("VERIFICATION REPORT")
    print("="*80)

    for new_pair_dir in new_pairs:
        try:
            # Construct OLD path
            rel_path = str(new_pair_dir).replace(NEW_EXP_BASE, "")
            # Ensure no leading slash issues if base paths don't match perfectly in ending slash
            if rel_path.startswith('/'): rel_path = rel_path[1:]
            
            old_pair_dir = os.path.join(OLD_EXP_BASE, rel_path)
            
            # Check if old pair exists in GT map
            if old_pair_dir not in gt_map:
                print(f"[SKIP] GT not found for {rel_path}")
                print(f"       Looking for Key: {old_pair_dir}")
                continue
                
            gt_info = gt_map[old_pair_dir]
            gt_indices = gt_info.get("gt_idx", [])
            
            if not gt_indices:
                print(f"[SKIP] No GT indices defined for {rel_path}")
                continue
            
            # Load OLD action options to get actual GT actions
            old_options_path = os.path.join(old_pair_dir, "action_options.json")
            if not os.path.exists(old_options_path):
                 # Try finding it if filename differs? usually reference is constant
                 print(f"[ERR ] Old action_options.json not found at {old_options_path}")
                 error_count += 1
                 continue
                 
            with open(old_options_path, 'r') as f:
                old_actions = json.load(f)
            
            gt_actions = [old_actions[i] for i in gt_indices if i < len(old_actions)]
            
            # Load NEW action options
            new_options_path = os.path.join(new_pair_dir, "action_options.json")
            with open(new_options_path, 'r') as f:
                new_actions = json.load(f)
            
            # Check for coverage
            # We want to see if AT LEAST ONE of the GT actions is present in the new set
            # Or should we check if *the* correct action is there?
            # Assuming if gt_indices defines "valid" actions, we want our new set to contain at least one valid action.
            
            # Let's normalize
            norm_gt = set(normalize_action(a) for a in gt_actions)
            norm_new = set(normalize_action(a) for a in new_actions)
            
            # Intersection
            intersection = norm_gt.intersection(norm_new)
            
            if intersection:
                print(f"[PASS] {rel_path}")
                # print(f"       Found {len(intersection)} matching GT action(s).")
                found_count += 1
            else:
                print(f"[FAIL] {rel_path}")
                print(f"       GT Actions: {norm_gt}")
                print(f"       New Actions: {norm_new}")
                missing_count += 1
                
        except Exception as e:
            print(f"[ERR ] Exception processing {new_pair_dir}: {e}")
            error_count += 1

    print("\n" + "="*80)
    print(f"Total: {len(new_pairs)}")
    print(f"Passed (GT exists): {found_count}")
    print(f"Failed (GT missing): {missing_count}")
    print(f"Errors: {error_count}")
    print("="*80)

if __name__ == "__main__":
    main()
