import logging
import os
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)

ROOT_DIR = os.environ.get("ROOT_DIR", "data/agent_eval_result/exp_2")

for dirpath, dirnames, filenames in tqdm(os.walk(ROOT_DIR)):
    if "action_options.json" in filenames:
        old_path = os.path.join(dirpath, "action_options.json")
        raw_path = os.path.join(dirpath, "action_options_raw.json")

        # 1️⃣ 重命名 action_option.json → action_option_raw.json
        os.rename(old_path, raw_path)

        # 2️⃣ 读取 raw 文件
        with open(raw_path, "r", encoding="utf-8") as f:
            action_option_raw = json.load(f)

        # 3️⃣ 提取 tool_call
        action_option = [
            action["tool_call"]
            for action in action_option_raw
            if "tool_call" in action
        ]

        # 4️⃣ 写回新的 action_option.json
        with open(old_path, "w", encoding="utf-8") as f:
            json.dump(action_option, f, indent=2, ensure_ascii=False)

        logger.info("Processed: %s", dirpath)
