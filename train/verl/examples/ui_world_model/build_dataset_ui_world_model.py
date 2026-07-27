import re
import json
import argparse
from pathlib import Path
from PIL import Image
import os
import datasets

_ACTION_RE = re.compile(r"-\s*Action:\s*(\{.*?\})\s*\n", re.DOTALL)

def parse_action_from_user_text(user_text: str):
    m = _ACTION_RE.search(user_text)
    if not m:
        idx = user_text.find("Action:")
        if idx == -1:
            return None
        s = user_text[idx:]
        brace = s.find("{")
        if brace == -1:
            return None
        s = s[brace:]
        depth = 0
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = s[: i + 1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        return None
        return None

    cand = m.group(1).strip()
    try:
        return json.loads(cand)
    except Exception:
        return None

def find_next_from_prev(prev_path: str, next_name="next.png"):
    p = Path(prev_path).expanduser()
    cand = p.with_name(next_name)
    return str(cand.resolve()) if cand.exists() else None

def read_bytes(p: str) -> bytes:
    with open(p, "rb") as f:
        return f.read()

def infer_split_from_json_path(sft_json_path: str, explicit: str | None):
    if explicit is not None:
        return explicit

    name = Path(sft_json_path).name.lower()
    if "train" in name:
        return "train"
    if "test" in name or "eval" in name or "val" in name or "valid" in name:
        return "test"
    raise ValueError(
        f"Cannot infer split from file name: {name}. "
        f"Please pass --split train|test explicitly."
    )

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--sft_json", default="/path/to/cuwm/project/office-wm/data/train_data_1231_2867.json")
    ap.add_argument("--sft_json", default="/path/to/cuwm/project/office-wm/data/test_data_1231_339.json")
    # ap.add_argument("--sft_json", default="/path/to/cuwm/project/office-wm/data/train_data_0131_991.json")
    ap.add_argument("--local_save_dir", default="/path/to/cuwm/project/rl-0.6.0/verl/examples/ui_world_model")
    ap.add_argument("--data_source", default="ui_world_model_rl")
    ap.add_argument("--next_name", default="next.png")
    ap.add_argument("--split", default=None, choices=["train", "test"],
                    help="可选：手动指定 split；不填则从 sft_json 文件名推断")
    args = ap.parse_args()

    split = infer_split_from_json_path(args.sft_json, args.split)

    data = json.load(open(args.sft_json, "r", encoding="utf-8"))
    rows = []
    skipped = {"no_prev": 0, "no_action": 0, "no_assistant": 0, "no_next": 0, "bad_image": 0}

    for idx, item in enumerate(data):
        imgs = item.get("images", [])
        if not imgs:
            skipped["no_prev"] += 1
            continue
        prev_path = imgs[0]
        
        pair_path = os.path.dirname(prev_path)

        msgs = item.get("messages", [])
        user = next((m for m in msgs if m.get("role") == "user"), None)
        asst = next((m for m in msgs if m.get("role") == "assistant"), None)

        if not asst or not asst.get("content", "").strip():
            skipped["no_assistant"] += 1
            continue
        assistant_text = asst["content"].strip()

        if not user:
            skipped["no_action"] += 1
            continue
        with open(os.path.join(pair_path, "action.json"), "r", encoding="utf-8") as f:
            groundtruth_action = json.load(f)
            print(groundtruth_action)
            
        if groundtruth_action is None:
            skipped["no_action"] += 1
            continue

        next_path = find_next_from_prev(prev_path, args.next_name)
        if not next_path:
            skipped["no_next"] += 1
            continue

        try:
            prev_img = Image.open(prev_path).convert("RGBA")
        except Exception:
            skipped["bad_image"] += 1
            continue

        rows.append({
            "data_source": args.data_source,
            # ✅ prompt 直接沿用 sft.json 的 messages
            "prompt": [user],
            "images": [prev_img],
            "ability": "office",
            "reward_model": {"style": "rule", "ground_truth": assistant_text},
            "extra_info":{
                "index": idx,
                "split": split,
                "groundtruth_action": json.dumps(groundtruth_action),
                "assistant_text": assistant_text,
                "pair_path": os.path.dirname(prev_path),
                "a11y_path": os.path.join(os.path.dirname(prev_path), "a11y.json"),
                "prev_path": prev_path,
                "next_path": next_path,
            }
        })
    import pdb;pdb.set_trace()
    out_dir = Path(args.local_save_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = datasets.Dataset.from_list(rows)
    out_path = out_dir / f"{split}_1k.parquet"
    ds.to_parquet(str(out_path))

    print(f"[OK] split={split} n={len(rows)} -> {out_path}")
    print("[SKIP]", skipped)

if __name__ == "__main__":
    main()
