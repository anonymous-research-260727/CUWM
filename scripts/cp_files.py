import logging
import os
import shutil

logger = logging.getLogger(__name__)

SRC_ROOT = os.environ.get("SRC_ROOT", "data/agent_eval_result/exp_2")
DST_ROOT = os.environ.get("DST_ROOT", "data/agent_eval_result/exp_dumb_selection_agent")

for dirpath, dirnames, filenames in os.walk(SRC_ROOT):
    if "action_options_raw.json" in filenames:
        src_file = os.path.join(dirpath, "action_options_raw.json")

        # 计算相对路径（从 1224_exp 开始）
        rel_path = os.path.relpath(src_file, SRC_ROOT)

        # 对应到 1224_exp_2
        dst_file = os.path.join(DST_ROOT, rel_path)

        # 确保目标目录存在
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)

        # 拷贝文件（覆盖）
        shutil.copy2(src_file, dst_file)

        logger.info("Copied: %s -> %s", src_file, dst_file)
