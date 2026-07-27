import logging
import os
import shutil

logger = logging.getLogger(__name__)

SRC_ROOT = os.environ.get("SRC_ROOT", "data/agent_eval_result/exp/new_prompt/test")
DST_ROOT = os.environ.get("DST_ROOT", "data/agent_eval_result/exp/new_prompt/test_mini")


for root, dirs, files in os.walk(SRC_ROOT):
    # print(f"Visiting: {root}")
    if "action_options.json" not in files:
        continue

    src_dir = root
    dst_dir = src_dir.replace(SRC_ROOT, DST_ROOT, 1)
    logger.info("Checking DST dir: %s", dst_dir)

    if not os.path.isdir(dst_dir):
        continue

    logger.info("Replacing:\n  SRC: %s\n  DST: %s", src_dir, dst_dir)
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    # 删除原 test_mini 中的目录
    # shutil.rmtree(src_dir)

    # 从 test 中拷贝同名目录过来
    # shutil.copytree(dst_dir, src_dir)