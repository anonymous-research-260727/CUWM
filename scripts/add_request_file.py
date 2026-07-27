import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_data_root = os.environ.get("DATA_ROOT", "data")
PAIRED_ROOTS = [
    Path(_data_root) / "test",
    Path(_data_root) / "test_mini",
]

SUCCESS_ROOT = Path(os.environ.get("SOURCE_DATA_ROOT", "data/test/data"))

PAIR_DIR_RE = re.compile(r"^pair_(\d+)$")

def read_nth_jsonl_request(jsonl_path: Path, n_zero_based: int) -> Optional[str]:
    """读取 jsonl 第 n_zero_based 行并返回 request。行不存在/解析失败则返回 None。"""
    if n_zero_based < 0:
        return None
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == n_zero_based:
                    line = line.strip()
                    if not line:
                        return None
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        return None
                    req = obj.get("request")
                    return req if isinstance(req, str) and req else None
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return None

def iter_pair_dirs(root: Path):
    """遍历 root 下所有形如 */*/paired/*/pair_XX 的目录，返回 pair_dir 路径。"""
    # 结构：root/{app}/{task}/paired/{name}/pair_{idx}
    # 用 glob 不写死 app/task/name
    for pair_dir in root.glob("*/*/paired/*/pair_*"):
        if pair_dir.is_dir():
            yield pair_dir

def main():
    total = 0
    written = 0
    skipped = 0

    for root in PAIRED_ROOTS:
        if not root.exists():
            logger.info("[skip] paired root not found: %s", root)
            continue

        for pair_dir in iter_pair_dirs(root):
            total += 1

            m = PAIR_DIR_RE.match(pair_dir.name)
            if not m:
                skipped += 1
                continue

            idx = int(m.group(1))              # pair_01 => 1
            line_no = idx - 1                  # 0-based
            name = pair_dir.parent.name        # {name}
            paired_dir = pair_dir.parent       # .../paired/{name}
            task_dir = paired_dir.parent.parent  # .../{task}
            app_dir = task_dir.parent            # .../{app}

            app = app_dir.name
            task = task_dir.name

            src_jsonl = SUCCESS_ROOT / app / task / "success" / f"{name}.jsonl"
            request = read_nth_jsonl_request(src_jsonl, line_no)

            if not request:
                logger.warning("[warn] missing request: pair_dir=%s  src=%s  line=%d", pair_dir, src_jsonl, line_no)
                skipped += 1
                continue

            out_path = pair_dir / "request.txt"
            out_path.write_text(request, encoding="utf-8")
            # print(f"[info] wrote request: {out_path}")
            # assert 0
            written += 1

    logger.info("Done. total_pair_dirs=%d, written=%d, skipped=%d", total, written, skipped)

if __name__ == "__main__":
    main()
