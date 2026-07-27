import re
import json
import argparse
from pathlib import Path
from PIL import Image

import datasets

# 源路径列表
source_paths = [
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4_2306/pair_02",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4_2306/pair_03",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4_2480/pair_02",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4_7037/pair_01",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4_7037/pair_02",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4_7037/pair_03",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4_7037/pair_04",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4s_1650/pair_02",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4s_1650/pair_04",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4s_1650/pair_06",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4s_1650/pair_07",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4s_1650/pair_08",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4s_3917/pair_02",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4s_3917/pair_03",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4s_5595/pair_01",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4s_6241/pair_03",
    "/path/to/cuwm/data_sample_3000/test/excel/bing_search/paired/excel_4s_9723/pair_02",
    "/path/to/cuwm/data_sample_3000/test/excel/m365/paired/excel_3_373/pair_01",
    "/path/to/cuwm/data_sample_3000/test/excel/m365/paired/excel_3_373/pair_02",
    "/path/to/cuwm/data_sample_3000/test/excel/m365/paired/excel_3_373/pair_03",
    "/path/to/cuwm/data_sample_3000/test/excel/m365/paired/excel_3_373/pair_04",
    "/path/to/cuwm/data_sample_3000/test/excel/m365/paired/excel_3_373/pair_05",
    "/path/to/cuwm/data_sample_3000/test/excel/m365/paired/excel_3_373/pair_08",
    "/path/to/cuwm/data_sample_3000/test/excel/m365/paired/excel_3_1334/pair_01",
    "/path/to/cuwm/data_sample_3000/test/excel/m365/paired/excel_3_1334/pair_02",
    "/path/to/cuwm/data_sample_3000/test/excel/m365/paired/excel_3_1334/pair_03",
    "/path/to/cuwm/data_sample_3000/test/excel/m365/paired/excel_3_1334/pair_04",
    "/path/to/cuwm/data_sample_3000/test/excel/qabench/paired/excel_1_2/pair_02",
    "/path/to/cuwm/data_sample_3000/test/excel/qabench/paired/excel_1_2/pair_03",
    "/path/to/cuwm/data_sample_3000/test/excel/qabench/paired/excel_1_101/pair_02",
    "/path/to/cuwm/data_sample_3000/test/excel/qabench/paired/excel_1_101/pair_03"
]

source_paths += [
    "/path/to/cuwm/data_sample_3000/test/ppt/bing_search/paired/ppt_4_1287/pair_01",
    "/path/to/cuwm/data_sample_3000/test/ppt/bing_search/paired/ppt_4_1287/pair_02",
    "/path/to/cuwm/data_sample_3000/test/ppt/bing_search/paired/ppt_4_7041/pair_01",
    "/path/to/cuwm/data_sample_3000/test/ppt/bing_search/paired/ppt_4_7041/pair_03",
    "/path/to/cuwm/data_sample_3000/test/ppt/bing_search/paired/ppt_4_7041/pair_04",
    "/path/to/cuwm/data_sample_3000/test/ppt/bing_search/paired/ppt_4_7041/pair_05",
    "/path/to/cuwm/data_sample_3000/test/ppt/bing_search/paired/ppt_4s_303/pair_03",
    "/path/to/cuwm/data_sample_3000/test/ppt/bing_search/paired/ppt_4s_303/pair_06",
    "/path/to/cuwm/data_sample_3000/test/ppt/bing_search/paired/ppt_4s_303/pair_07",
    "/path/to/cuwm/data_sample_3000/test/ppt/bing_search/paired/ppt_4s_303/pair_09",
    "/path/to/cuwm/data_sample_3000/test/ppt/bing_search/paired/ppt_4s_6747/pair_04",
    "/path/to/cuwm/data_sample_3000/test/ppt/bing_search/paired/ppt_4s_9578/pair_01",
    "/path/to/cuwm/data_sample_3000/test/ppt/bing_search/paired/ppt_4s_9578/pair_02",
    "/path/to/cuwm/data_sample_3000/test/ppt/m365/paired/ppt_3_110/pair_02",
    "/path/to/cuwm/data_sample_3000/test/ppt/m365/paired/ppt_3_136/pair_02",
    "/path/to/cuwm/data_sample_3000/test/ppt/m365/paired/ppt_3_185/pair_01",
    "/path/to/cuwm/data_sample_3000/test/ppt/m365/paired/ppt_3_185/pair_02",
    "/path/to/cuwm/data_sample_3000/test/ppt/m365/paired/ppt_3_233/pair_01",
    "/path/to/cuwm/data_sample_3000/test/ppt/m365/paired/ppt_3_233/pair_02",
    "/path/to/cuwm/data_sample_3000/test/ppt/m365/paired/ppt_3_636/pair_01",
    "/path/to/cuwm/data_sample_3000/test/ppt/m365/paired/ppt_3_636/pair_02",
    "/path/to/cuwm/data_sample_3000/test/ppt/m365/paired/ppt_3_636/pair_04",
    "/path/to/cuwm/data_sample_3000/test/ppt/qabench/paired/ppt_1_36/pair_01",
    "/path/to/cuwm/data_sample_3000/test/ppt/qabench/paired/ppt_1_49/pair_01",
    "/path/to/cuwm/data_sample_3000/test/ppt/qabench/paired/ppt_1_49/pair_04",
    "/path/to/cuwm/data_sample_3000/test/ppt/qabench/paired/ppt_1_92/pair_02",
    "/path/to/cuwm/data_sample_3000/test/ppt/qabench/paired/ppt_1_173/pair_01",
    "/path/to/cuwm/data_sample_3000/test/word/bing_search/paired/word_4_369/pair_01",
    "/path/to/cuwm/data_sample_3000/test/word/bing_search/paired/word_4_369/pair_03",
    "/path/to/cuwm/data_sample_3000/test/word/bing_search/paired/word_4_369/pair_05",
    "/path/to/cuwm/data_sample_3000/test/word/bing_search/paired/word_4_8820/pair_02",
    "/path/to/cuwm/data_sample_3000/test/word/bing_search/paired/word_4_10199/pair_04",
    "/path/to/cuwm/data_sample_3000/test/word/bing_search/paired/word_4_14485/pair_01",
    "/path/to/cuwm/data_sample_3000/test/word/bing_search/paired/word_4_19965/pair_02",
    "/path/to/cuwm/data_sample_3000/test/word/bing_search/paired/word_4s_2144/pair_04",
    "/path/to/cuwm/data_sample_3000/test/word/bing_search/paired/word_4s_2144/pair_05",
    "/path/to/cuwm/data_sample_3000/test/word/bing_search/paired/word_4s_9251/pair_02",
    "/path/to/cuwm/data_sample_3000/test/word/bing_search/paired/word_4s_15379/pair_01",
    "/path/to/cuwm/data_sample_3000/test/word/bing_search/paired/word_4s_15379/pair_03",
    "/path/to/cuwm/data_sample_3000/test/word/bing_search/paired/word_4s_15379/pair_04",
    "/path/to/cuwm/data_sample_3000/test/word/m365/paired/word_3_30/pair_01",
    "/path/to/cuwm/data_sample_3000/test/word/m365/paired/word_3_30/pair_02",
    "/path/to/cuwm/data_sample_3000/test/word/m365/paired/word_3_39/pair_01",
    "/path/to/cuwm/data_sample_3000/test/word/m365/paired/word_3_93/pair_01",
    "/path/to/cuwm/data_sample_3000/test/word/m365/paired/word_3_93/pair_02",
    "/path/to/cuwm/data_sample_3000/test/word/m365/paired/word_3_93/pair_03",
    "/path/to/cuwm/data_sample_3000/test/word/m365/paired/word_3_93/pair_04",
    "/path/to/cuwm/data_sample_3000/test/word/m365/paired/word_3_563/pair_01",
    "/path/to/cuwm/data_sample_3000/test/word/m365/paired/word_3_604/pair_01",
    "/path/to/cuwm/data_sample_3000/test/word/m365/paired/word_3_604/pair_03",
    "/path/to/cuwm/data_sample_3000/test/word/m365/paired/word_3_650/pair_01",
    "/path/to/cuwm/data_sample_3000/test/word/m365/paired/word_3_757/pair_02",
    "/path/to/cuwm/data_sample_3000/test/word/m365/paired/word_3_757/pair_03",
    "/path/to/cuwm/data_sample_3000/test/word/qabench/paired/word_1_1/pair_02",
    "/path/to/cuwm/data_sample_3000/test/word/qabench/paired/word_1_105/pair_01",
    "/path/to/cuwm/data_sample_3000/test/word/qabench/paired/word_1_105/pair_02",
    "/path/to/cuwm/data_sample_3000/test/word/qabench/paired/word_1_221/pair_02",
    "/path/to/cuwm/data_sample_3000/test/word/qabench/paired/word_1_221/pair_03",
    "/path/to/cuwm/data_sample_3000/test/word/qabench/paired/word_1_221/pair_04",
    "/path/to/cuwm/data_sample_3000/test/word/qabench/paired/word_1_221/pair_05",
    "/path/to/cuwm/data_sample_3000/test/word/wikihow/paired/word_2_147/pair_01",
    "/path/to/cuwm/data_sample_3000/test/word/wikihow/paired/word_2_147/pair_03",
    "/path/to/cuwm/data_sample_3000/test/word/wikihow/paired/word_2_151/pair_01",
    "/path/to/cuwm/data_sample_3000/test/word/wikihow/paired/word_2_151/pair_03",
    "/path/to/cuwm/data_sample_3000/test/word/wikihow/paired/word_2_165/pair_02",
    "/path/to/cuwm/data_sample_3000/test/word/wikihow/paired/word_2_165/pair_03",
    "/path/to/cuwm/data_sample_3000/test/word/wikihow/paired/word_2_165/pair_06"
]

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

# ✅ 新增：把 /.../test/.../paired/.../pair_xx/prev.png 归一化成 /test/.../paired/.../pair_xx
def _extract_pair_key_from_prev(prev_path: str) -> str | None:
    """
    Return normalized key like:
      test/word/m365/paired/word_3_93/pair_19
    from an absolute path ending with .../test/.../paired/.../pair_xx/prev.png
    """
    p = Path(prev_path).expanduser().resolve()
    parts = p.parts  # tuple of path components
    try:
        test_idx = parts.index("test")
    except ValueError:
        return None

    # find the first component after 'test' that matches 'pair_XX'
    pair_idx = None
    for i in range(test_idx + 1, len(parts)):
        if re.fullmatch(r"pair_\d+", parts[i]):
            pair_idx = i
            break
    if pair_idx is None:
        return None

    key_parts = parts[test_idx : pair_idx + 1]  # include 'test' ... 'pair_XX'
    return "/".join(key_parts)


def _build_allowed_pair_keys(source_paths_list) -> set[str]:
    keys = set()
    for sp in source_paths_list:
        p = Path(sp).expanduser().resolve()
        parts = p.parts
        try:
            test_idx = parts.index("test")
        except ValueError:
            # If a source_path doesn't contain 'test', skip it
            continue
        # source path itself ends at pair_XX, so we keep from test..pair
        key = "/".join(parts[test_idx:])
        keys.add(key)
    return keys

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--sft_json", default="/path/to/cuwm/project/office-wm/data/train_data_1231_2867.json")
    ap.add_argument("--sft_json", default="/path/to/cuwm/project/office-wm/data/test_data_1231_339.json")
    ap.add_argument("--local_save_dir", default="/path/to/cuwm/project/rl-0.6.0/verl/examples/data_preprocess")
    ap.add_argument("--data_source", default="ui_world_model_rl")
    ap.add_argument("--next_name", default="next.png")
    ap.add_argument("--split", default=None, choices=["train", "test"],
                    help="可选：手动指定 split；不填则从 sft_json 文件名推断")
    args = ap.parse_args()

    split = infer_split_from_json_path(args.sft_json, args.split)
    # ✅ 新增：构建允许的 pair key 集合
    allowed_pair_keys = _build_allowed_pair_keys(source_paths)

    data = json.load(open(args.sft_json, "r", encoding="utf-8"))
    rows = []
    skipped = {
        "no_prev": 0,
        "no_action": 0,
        "no_assistant": 0,
        "no_next": 0,
        "bad_image": 0,
        "not_in_source_paths": 0,  # ✅ 新增
        "bad_prev_path": 0,        # ✅ 新增（提取 key 失败）
    }


    for idx, item in enumerate(data):
        imgs = item.get("images", [])
        if not imgs:
            skipped["no_prev"] += 1
            continue
        prev_path = imgs[0]

        # ✅ 新增：只保留 prev_path 对应的 test/.../pair_xx 出现在 source_paths 中的样本
        pair_key = _extract_pair_key_from_prev(prev_path)
        if pair_key is None:
            skipped["bad_prev_path"] += 1
            continue
        if pair_key not in allowed_pair_keys:
            skipped["not_in_source_paths"] += 1
            continue

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
        action_obj = parse_action_from_user_text(user.get("content", ""))
        if action_obj is None:
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

        # try:
        #     next_img = Image.open(next_path).convert("RGBA")
        # except Exception:
        #     skipped["bad_image"] += 1
        #     continue

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
                "action": json.dumps(action_obj),
                "assistant_text": assistant_text,
            }
        })

    out_dir = Path(args.local_save_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = datasets.Dataset.from_list(rows)
    out_path = out_dir / f"{split}.parquet"
    ds.to_parquet(str(out_path))

    print(f"[OK] split={split} n={len(rows)} -> {out_path}")
    print("[SKIP]", skipped)

if __name__ == "__main__":
    main()
