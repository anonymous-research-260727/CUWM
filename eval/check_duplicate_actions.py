#!/usr/bin/env python3
"""
Script to check for duplicate predicted actions in pairs.

For each pair's action_options.json, checks if all actions are unique.
Reports pairs that have duplicate actions.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

ROOT = os.environ.get("DATA_ROOT", "data")
EXP_NAME = "0128_gemini_qwen_sft"  # Change this to your experiment name


def normalize_action(action):
    """
    Normalize an action to a comparable format.
    Converts args dict to sorted JSON string for comparison.
    """
    normalized = {
        "function": action.get("function", ""),
        "args": json.dumps(action.get("args", {}), sort_keys=True),
        "status": action.get("status", "")
    }
    return json.dumps(normalized, sort_keys=True)


def check_duplicates_in_pair(action_options_path):
    """
    Check if there are duplicate actions in a pair's action_options.json.
    
    Returns:
        tuple: (has_duplicates: bool, duplicates_info: list)
    """
    try:
        with open(action_options_path, 'r', encoding='utf-8') as f:
            actions = json.load(f)
        
        if not isinstance(actions, list):
            return True, [f"Invalid format: not a list"]
        
        # Normalize all actions
        normalized_actions = [normalize_action(action) for action in actions]
        
        # Find duplicates
        seen = {}
        duplicates = []
        
        for idx, norm_action in enumerate(normalized_actions):
            if norm_action in seen:
                duplicates.append({
                    "indices": [seen[norm_action], idx],
                    "action": actions[idx]
                })
            else:
                seen[norm_action] = idx
        
        return len(duplicates) > 0, duplicates
        
    except Exception as e:
        return True, [f"Error reading file: {e}"]


def find_all_pairs(base_dir):
    """Find all pair directories that have action_options.json"""
    pairs = []
    base_path = Path(base_dir)
    
    for action_options_path in base_path.rglob("action_options.json"):
        pair_dir = action_options_path.parent
        pairs.append({
            "path": str(action_options_path),
            "pair_dir": str(pair_dir),
            "relative_path": str(action_options_path.relative_to(base_path))
        })
    
    return pairs


def main():
    base_dir = os.path.join(ROOT, "agent_eval_result", EXP_NAME)
    
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return
    
    print(f"Scanning for action_options.json files in: {base_dir}")
    pairs = find_all_pairs(base_dir)
    print(f"Found {len(pairs)} pairs with action_options.json\n")
    
    pairs_with_duplicates = []
    pairs_without_duplicates = []
    pairs_with_errors = []
    
    for pair_info in pairs:
        has_duplicates, duplicates_info = check_duplicates_in_pair(pair_info["path"])
        
        if duplicates_info and isinstance(duplicates_info[0], str) and "Error" in duplicates_info[0]:
            pairs_with_errors.append({
                "pair": pair_info["relative_path"],
                "error": duplicates_info[0]
            })
        elif has_duplicates:
            pairs_with_duplicates.append({
                "pair": pair_info["relative_path"],
                "duplicates": duplicates_info
            })
        else:
            pairs_without_duplicates.append(pair_info["relative_path"])
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total pairs checked: {len(pairs)}")
    print(f"Pairs with duplicates: {len(pairs_with_duplicates)}")
    print(f"Pairs without duplicates: {len(pairs_without_duplicates)}")
    print(f"Pairs with errors: {len(pairs_with_errors)}")
    
    # Print pairs with duplicates
    if pairs_with_duplicates:
        print("\n" + "=" * 80)
        print("PAIRS WITH DUPLICATE ACTIONS")
        print("=" * 80)
        for pair_info in pairs_with_duplicates:
            print(f"\n{pair_info['pair']}:")
            for dup in pair_info['duplicates']:
                print(f"  Duplicate found at indices: {dup['indices']}")
                print(f"  Action: {json.dumps(dup['action'], indent=4)}")
    
    # Print pairs with errors
    if pairs_with_errors:
        print("\n" + "=" * 80)
        print("PAIRS WITH ERRORS")
        print("=" * 80)
        for pair_info in pairs_with_errors:
            print(f"{pair_info['pair']}: {pair_info['error']}")
    
    # Save results to file
    output_file = os.path.join(base_dir, "duplicate_actions_report.json")
    report = {
        "summary": {
            "total_pairs": len(pairs),
            "pairs_with_duplicates": len(pairs_with_duplicates),
            "pairs_without_duplicates": len(pairs_without_duplicates),
            "pairs_with_errors": len(pairs_with_errors)
        },
        "pairs_with_duplicates": pairs_with_duplicates,
        "pairs_with_errors": pairs_with_errors
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed report saved to: {output_file}")


if __name__ == "__main__":
    main()

