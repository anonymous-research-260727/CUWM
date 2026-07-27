#!/bin/bash

# 需要软链的模型列表
declare -A model_map=(
    ["Qwen-Image-Edit-2509"]="models--Qwen--Qwen-Image-Edit-2509/snapshots/d3968ef930e841f4c73640fb8afa3b306a78167e"
    ["Qwen-Image"]="models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6"
    # ["Qwen-Image-Edit"]="models--Qwen--Qwen-Image-Edit/snapshots/<在此填写你的 snapshot 哈希>"
)

# 根路径
target_root="/path/to/local_storage/DiffSynth-Studio/models/Qwen"
source_root="/path/to/local_storage/hub"

echo "开始创建软链接..."

for model in "${!model_map[@]}"; do
    source_path="$source_root/${model_map[$model]}"
    target_path="$target_root/$model"

    echo "处理模型：$model"
    echo "源目录：$source_path"
    echo "目标目录：$target_path"

    if [ ! -d "$source_path" ]; then
        echo "❌ 未找到源目录，跳过：$source_path"
        continue
    fi

    # 遍历源目录下所有子文件夹
    find "$source_path" -type d | while read -r src_subdir; do
        # 计算相对路径
        rel_path="${src_subdir#$source_path/}"
        tgt_subdir="$target_path/$rel_path"

        # 创建目标子目录
        mkdir -p "$tgt_subdir"

        # 查找所有 safetensors 文件
        for file in "$src_subdir"/*.safetensors; do
            if [ -f "$file" ]; then
                ln -sf "$file" "$tgt_subdir/$(basename "$file")"
                echo "  ✔ 链接：$file -> $tgt_subdir/"
            fi
        done
    done
done

echo "全部软链接创建完成！"
