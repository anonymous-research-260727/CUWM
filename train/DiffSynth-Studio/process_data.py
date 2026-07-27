import os
import json

def collect_data(root_dir, output_json="gui_qabench_data_191.json"):
    results = []

    for root, dirs, files in os.walk(root_dir):
        # 检查必要文件是否存在
        if "prev.png" in files and "next.png" in files and "qwen_prompt.txt" in files:
            prev_path = os.path.join(root, "prev.png")
            next_path = os.path.join(root, "next.png")
            prompt_path = os.path.join(root, "qwen_prompt.txt")

            # 读取 prompt 内容
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()

            # 生成相对路径（可选：如果你要保持根目录下的相对结构）
            rel_prev = os.path.relpath(prev_path, root_dir)
            rel_next = os.path.relpath(next_path, root_dir)

            # 组装数据
            item = {
                "image": rel_next.replace("\\", "/"),   # 转成 unix 风格
                "prompt": prompt_text,
                "edit_image": rel_prev.replace("\\", "/")
            }
            results.append(item)

    # 写成 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return results


# 示例调用
if __name__ == "__main__":
    data = collect_data("data_sample_3000", output_json="data_sample_3000.json")  # 你的根目录
    print("生成数据数量:", len(data))
    print("示例:", data[0] if data else "无结果")
