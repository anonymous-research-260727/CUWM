import os
import re
import shutil
import json
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image
import cv2


def extract_number(s: str) -> int:
    """从文件名中提取第一个数字，用于自然排序."""
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else -1


prompt = """Based on the previous UI screenshot and the described action, generate the next realistic frame showing the updated screen.
action：{action}
"""


def images_equal(path1: str, path2: str) -> bool:
    """判断两张图片像素级是否完全一样."""
    try:
        im1 = Image.open(path1).convert("RGB")
        im2 = Image.open(path2).convert("RGB")
    except Exception as e:
        # 如果图片打不开，保守起见认为不一样
        print(f"[WARN] fail to open image: {path1} or {path2}, err={e}")
        return False

    if im1.size != im2.size:
        return False

    arr1 = np.array(im1)
    arr2 = np.array(im2)
    return np.array_equal(arr1, arr2)

def detect_black_borders(img_path, threshold=0):
    """
    检测图片四周的黑边区域。
    threshold：像素亮度阈值，小于此值可认为接近黑色
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    def is_black_line(line):
        return np.mean(line) <= threshold

    top = 0
    for i in range(h):
        if is_black_line(gray[i]):
            top = i
        else:
            break

    bottom = h - 1
    for i in range(h - 1, -1, -1):
        if is_black_line(gray[i]):
            bottom = i
        else:
            break

    left = 0
    for j in range(w):
        if is_black_line(gray[:, j]):
            left = j
        else:
            break

    right = w - 1
    for j in range(w - 1, -1, -1):
        if is_black_line(gray[:, j]):
            right = j
        else:
            break

    return top, h - bottom - 1, left, w - right - 1, h, w


def process_single_subdir(subdir: str, folder: str):
    """
    处理单个子目录，提取所有合法的图片对。
    
    Args:
        subdir: 子目录的完整路径
        folder: 子目录的名称（用作 session）
    
    Returns:
        list: 该子目录下所有合法的图片对信息
    """
    rows = []
    
    # 获取 diff/action 文件
    actions = [
        f for f in os.listdir(subdir)
        if f.startswith("action_") and f.endswith(".txt")
    ]
    actions.sort(key=extract_number)

    # 相邻图片配对 + 对应 diff
    for i in range(len(actions) - 1):
        prev_img = os.path.join(subdir, actions[i].replace("txt", "png"))
        next_img = os.path.join(subdir, actions[i + 1].replace("txt", "png"))
        action_path = os.path.join(subdir, actions[i]) if i < len(actions) else None
        # import pdb;pdb.set_trace()
        # 读 action 文本
        action_text_raw = ""
        if action_path and os.path.exists(action_path):
            try:
                with open(action_path, "r", encoding="utf-8") as f:
                    action_text_raw = f.read().strip()
            except Exception:
                action_text_raw = ""

        # action 为空直接跳过
        if action_text_raw == "":
            continue

        # 解析 JSON
        try:
            action_obj = json.loads(action_text_raw)
        except Exception as e:
            print(f"[WARN] fail to parse json: {action_path}, err={e}")
            continue

        # 无效 action 过滤
        if action_obj.get("control_name", "") == "" and \
           action_obj.get("function", "") == "" and \
           action_obj.get("args", {}) == {}:
            # print("[SKIP] empty action:", action_obj)
            continue

        # 图片完全一样就跳过
        if images_equal(prev_img, next_img):
            # print(f"[SKIP] identical images: {prev_img} vs {next_img}")
            continue

        if "x" in action_obj.get("args", {}) and "y" in action_obj.get("args", {}) and action_obj.get("args", {})["x"] is None and action_obj.get("args", {})["y"] is None:
            print("[SKIP] action with null coords:", action_obj)
            continue

        if action_obj.get("control_name", "") == "":
            # print("[SKIP] action with empty control_name:", action_obj)
            continue
        
        prev_top, prev_bottom, prev_left, prev_right, prev_h, prev_w = detect_black_borders(prev_img, threshold=0)
        next_top, next_bottom, next_left, next_right, next_h, next_w = detect_black_borders(next_img, threshold=0)
        
        prev_border = (prev_top, prev_bottom, prev_left, prev_right)
        next_border = (next_top, next_bottom, next_left, next_right)
        
        prev_imgsize = (prev_h, prev_w)
        next_imgsize = (next_h, next_w)
        
        if prev_border != (7, 0, 7, 7) or next_border != (7, 0, 7, 7) or prev_imgsize != (736, 1040) or next_imgsize != (736, 1040):
            print(f"[SKIP] black border or size mismatch: {prev_img} border={prev_border} size={prev_imgsize}, {next_img} border={next_border} size={next_imgsize}")
            continue
        
        action_args = action_obj.get("args", {})
        args_keys = set(action_args.keys())
    
        possible_xy_pairs = [
            ("x", "y"),
            ("start_x", "start_y"),
            ("end_x", "end_y"),
            ("desktop_start_x", "desktop_start_y"),
            ("desktop_end_x", "desktop_end_y"),
            ("desktop_x", "desktop_y"),
        ]
        x_y_location_flag = False
        for x_key, y_key in possible_xy_pairs:
            if x_key in args_keys or y_key in args_keys:
                x_val = action_args.get(x_key, None)
                y_val = action_args.get(y_key, None)
                if x_val is None or y_val is None:
                    x_y_location_flag = True
                    print(f"[SKIP] action with null coords: {action_obj}")
                    break
                if not (0 <= x_val <= 1040 and 0 <= y_val <= 736):
                    x_y_location_flag = True
                    print(f"[SKIP] action with out-of-bounds coords: {action_obj}")
                    break
            if x_y_location_flag:
                break
        if x_y_location_flag:
            continue

        # 这里只记录源路径和 action 信息，不复制文件
        rows.append({
            "src_prev": prev_img,
            "src_next": next_img,
            "src_action": action_path,
            "action_obj": action_obj,
            "session": folder,     # word_3_4 / excel_2_1 ...
            "pair_idx": i + 1,     # 原始对的下标，仅用于参考
        })
    
    return rows


def pair_images_with_diffs(src_root: str):
    """
    为单个任务目录（例如 data/train/image/word/bing_search/success）
    构造 prev/next/action 的成对数据，但不复制文件，只返回信息。
    rows 里额外记录 "session"（例如 word_3_4），方便后续按 session 抽样。
    """
    rows = []

    for folder in tqdm(sorted(os.listdir(src_root)), desc=f"Scanning folders in {src_root}"):
        subdir = os.path.join(src_root, folder)
        if not os.path.isdir(subdir):
            continue

        # 调用新的函数处理单个子目录
        subdir_rows = process_single_subdir(subdir, folder)
        rows.extend(subdir_rows)

    return rows


def sample_rows_by_session(rows, target=300, seed=42):
    """
    按 session 抽样：一旦选中某个 session，就把该 session 下所有 pair 一起选中，
    直到总样本数 >= target 或 session 用完。
    """
    import random
    random.seed(seed)

    groups = defaultdict(list)
    for r in rows:
        sid = r.get("session", "unknown")
        groups[sid].append(r)

    session_ids = list(groups.keys())
    random.shuffle(session_ids)

    selected = []
    for sid in session_ids:
        selected.extend(groups[sid])
        if len(selected) >= target:
            break

    return selected


if __name__ == "__main__":
    choice = "train"
    srcs = [
        "data/train/image/word/bing_search/success",
        "data/train/image/word/m365/success",
        "data/train/image/word/qabench/success",
        "data/train/image/word/wikihow/success",
        "data/train/image/excel/bing_search/success",
        "data/train/image/excel/m365/success",
        "data/train/image/excel/qabench/success",
        "data/train/image/ppt/bing_search/success",
        "data/train/image/ppt/m365/success",
        "data/train/image/ppt/qabench/success",
    ]

    # choice = "test"
    # srcs = [
    #     "data/test/image/word/bing_search/success",
    #     "data/test/image/word/m365/success",
    #     "data/test/image/word/qabench/success",
    #     "data/test/image/word/wikihow/success",
    #     "data/test/image/excel/bing_search/success",
    #     "data/test/image/excel/m365/success",
    #     "data/test/image/excel/qabench/success",
    #     "data/test/image/ppt/bing_search/success",
    #     "data/test/image/ppt/m365/success",
    #     "data/test/image/ppt/qabench/success",
    # ]

    all_rows_for_csv = []

    for src in srcs:
        # 使用正则提取 task 名（如 word/bing_search）
        match = re.match(rf"^data/{choice}/image/(.+?)/success$", src)
        if match:
            task_name = match.group(1)  # e.g. "word/bing_search"
            print("Task:", task_name)
        else:
            print(src, "未匹配")
            continue

        # 目标根目录（只会为“采样出来”的 pair 建目录/复制）
        dst_root = f"data_sample_10000/{choice}/{task_name}/paired"

        # 1) 先扫描这个任务的所有合法 pair（不复制文件）
        rows_all = pair_images_with_diffs(src)
        print(f"{task_name}: total valid pairs = {len(rows_all)}")

        # 2) 再按 session 维度抽样到 ~N（这里示例 target=40）
        rows_sampled = sample_rows_by_session(rows_all, target=200)
        print(f"{task_name}: sampled pairs = {len(rows_sampled)}")

        # 3) 只对采样出来的 rows 建目录 + 复制
        #    为了目录好看，这里对每个 session 单独编号 pair_01, pair_02, ...
        per_session_counter = defaultdict(int)

        for r in rows_sampled:
            session = r["session"]
            per_session_counter[session] += 1
            idx = per_session_counter[session]

            out_dir = os.path.join(dst_root, session, f"pair_{idx:02d}")
            os.makedirs(out_dir, exist_ok=True)

            # 复制 prev / next
            dst_prev = os.path.join(out_dir, "prev.png")
            dst_next = os.path.join(out_dir, "next.png")
            shutil.copy2(r["src_prev"], dst_prev)
            shutil.copy2(r["src_next"], dst_next)

            # 复制 / 写入 action.txt
            if r["src_action"] and os.path.exists(r["src_action"]):
                dst_action = os.path.join(out_dir, "action.txt")
                shutil.copy2(r["src_action"], dst_action)
            else:
                dst_action = os.path.join(out_dir, "action.txt")
                with open(dst_action, "w", encoding="utf-8") as f:
                    f.write("[WARN] action missing\n")

            # 生成 prompt 文本
            action_str = json.dumps(r["action_obj"], ensure_ascii=False)
            prompt_text = prompt.format(action=action_str)

            all_rows_for_csv.append({
                "image": dst_next,
                "prompt": prompt_text,
                "edit_image": dst_prev,
                "session": session,
            })
    # 生成data_sample_3000
    output_root = os.environ.get("DATA_ROOT", "data")
    out_csv_path = os.path.join(output_root, choice, f"guidata_edit_{choice}.csv")
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    df = pd.DataFrame(all_rows_for_csv, columns=["image", "prompt", "edit_image", "session"])
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    print(f"[OK] 汇总 {len(df)} 对，已保存到：{out_csv_path}")
