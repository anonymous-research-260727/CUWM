import os
import json

_output_dir = os.environ.get("OUTPUT_DIR", "output")

with open(os.path.join(_output_dir, "gpt-5.2-chat-20251211.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    folder = os.path.dirname(item["images"][0])
    pred = item["pred"]
    # Delete old prompt_nl_pred.txt if it exists
    old_pred_file = os.path.join(folder, "prompt_nl_pred.txt")
    if os.path.exists(old_pred_file):
        os.remove(old_pred_file)
        
    with open(os.path.join(folder, "prompt_nl_pred_gpt-5.2-chat-20251211.txt"), "w", encoding="utf-8") as f:
        f.write(pred)
        
    gt = item["gt"]
    with open(os.path.join(folder, "prompt_nl_gt.txt"), "w", encoding="utf-8") as f:
        f.write(gt)

with open(os.path.join(_output_dir, "checkpoint-450-merged.json"), encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    folder = os.path.dirname(item["images"][0])
    pred = item["pred"]
    with open(os.path.join(folder, "prompt_nl_pred_checkpoint-450-merged.txt"), "w", encoding="utf-8") as f:
        f.write(pred)
        
with open(os.path.join(_output_dir, "base.json"), encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    folder = os.path.dirname(item["images"][0])
    pred = item["pred"]
    with open(os.path.join(folder, "prompt_nl_pred_base.txt"), "w", encoding="utf-8") as f:
        f.write(pred)