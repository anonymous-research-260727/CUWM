# Computer-Using World Model — 训练与推理指南

基于 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 框架，对 **Qwen-Image-Edit-2509** 进行 LoRA 微调，使其能够根据文本 prompt 生成 GUI 操作后的下一帧截图。

---

## 目录结构

```
DiffSynth-Studio/
├── examples/qwen_image/model_training/
│   ├── train.py                          # 训练入口脚本
│   ├── lora/                             # 训练启动 shell 脚本 & 数据 CSV
│   │   ├── Qwen-Image-Edit-2509-prompt-1226-freezevl.sh   # ★ 主训练脚本（冻结 VL 编码器）
│   │   ├── Qwen-Image-Edit-2509.sh                        # 基线训练脚本
│   │   └── pairs_summary_data_sample_3000_train_prompt_1226.csv  # 训练数据索引
│   ├── validate_lora/                    # 推理/验证脚本
│   │   └── Qwen-Image-Edit-2509.py       # 通用 LoRA 验证入口
│   └── scripts/                          # 辅助初始化脚本
├── models/                               # 模型权重存放目录
│   └── Qwen/                             # 通过 soft-link.sh 链接到 HF 缓存
│       ├── Qwen-Image-Edit-2509/         # Transformer (DiT) 权重
│       ├── Qwen-Image/                   # Text Encoder + VAE 权重
│       └── Qwen-Image-Edit/              # Processor (tokenizer/feature extractor)
└── diffsynth/                            # DiffSynth 框架源码
```

---

## 模型位置

### 基础模型（预训练权重）

| 组件 | Model ID | 文件路径模式 | 说明 |
|------|----------|-------------|------|
| **Transformer (DiT)** | `Qwen/Qwen-Image-Edit-2509` | `transformer/diffusion_pytorch_model*.safetensors` | 扩散模型主干 |
| **Text Encoder** | `Qwen/Qwen-Image` | `text_encoder/model*.safetensors` | 文本编码器（冻结 VL 版本使用原始权重） |
| **VAE** | `Qwen/Qwen-Image` | `vae/diffusion_pytorch_model.safetensors` | 图像编解码器 |
| **Processor** | `Qwen/Qwen-Image-Edit` | `processor/` | Tokenizer & Feature Extractor |

### LoRA 检查点（训练输出）

- 输出路径：`models/train/Qwen-Image-Edit-2509_data_sample_3000_freezeqwenvl_lora_prompt1226/`
- 每个 epoch 保存一个 `epoch-{N}.safetensors` 文件
- 推理时从 `<YOUR_AZURE_BLOB_ROOT>/gym_project/gym_project_ckpt/Qwen-Image-Edit-2509_data_sample_3000_freezeqwenvl_lora_prompt1226_datarepeat10_merge/epoch-24.safetensors` 目录加载对应 epoch 的权重

---

## 数据位置与组织格式

### 数据根目录

- **训练/验证数据**：`<YOUR_AZURE_BLOB_ROOT>/gym_project/yr-data-sample-3000/data_sample_3000/new_prompt/train/` 和 `<YOUR_AZURE_BLOB_ROOT>/gym_project/yr-data-sample-3000/data_sample_3000/new_prompt/test/`
- **训练索引 CSV**：`examples/qwen_image/model_training/lora/pairs_summary_data_sample_3000_train_prompt_1226.csv`

### 数据组织格式

数据按照 **应用类型 → 数据来源 → paired → 样本 → pair** 的层次结构组织：

```
data_sample_3000/
├── train_valid/                     # 训练验证集
│   ├── word/
│   │   ├── qabench/
│   │   │   └── paired/
│   │   │       ├── word_1_1/
│   │   │       │   ├── pair_01/
│   │   │       │   │   ├── prev.png                              # 操作前截图（输入）
│   │   │       │   │   ├── next.png                              # 操作后截图（GT）
│   │   │       │   │   ├── qwen_prompt.txt                       # GT prompt（GPT 生成）
│   │   │       │   │   ├── qwen_prompt_qwenvl_pred.txt           # QwenVL 预测 prompt
│   │   │       │   │   ├── qwen_prompt_qwenvl_lora_pred.txt      # QwenVL LoRA 预测 prompt
│   │   │       │   │   └── qwen_prompt_gpt5_pred.txt             # GPT-5 预测 prompt
│   │   │       │   ├── pair_02/
│   │   │       │   └── ...
│   │   │       └── word_1_103/
│   │   └── ...
│   ├── excel/
│   │   └── bing_search/paired/...
│   └── ppt/
│       └── bing_search/paired/...
├── test_mini_new_prompt/            # 测试集（结构同上）
└── ...
```

### 每个 pair 目录中的文件

| 文件名 | 说明 |
|--------|------|
| `prev.png` | 操作前的 GUI 截图（模型输入） |
| `next.png` | 操作后的 GUI 截图（Ground Truth） |
| `qwen_prompt.txt` | GT prompt — 由 GPT 生成的详细操作描述 |
| `qwen_prompt_qwenvl_pred.txt` | QwenVL base 模型预测的 prompt |
| `qwen_prompt_qwenvl_lora_pred.txt` | QwenVL LoRA 微调模型预测的 prompt |
| `qwen_prompt_gpt5_pred.txt` | GPT-5 预测的 prompt |
| `next_*_epoch_*.png` | 各 epoch / prompt 类型的推理结果（由推理脚本生成） |

### 训练数据 CSV 格式

CSV 文件包含三列，用于训练数据加载：

```csv
image,prompt,edit_image
word_qabench_191/data/word_1_103/pair_01/next.png,"Generate the next frame of ...",word_qabench_191/data/word_1_103/pair_01/prev.png
```

| 列名 | 说明 |
|------|------|
| `image` | 目标图像路径（next.png，即 Ground Truth） |
| `prompt` | 操作描述文本 |
| `edit_image` | 输入图像路径（prev.png，即操作前截图） |

路径为相对 `--dataset_base_path`的相对路径。

---

## 训练

### 训练脚本

主训练脚本：`examples/qwen_image/model_training/lora/Qwen-Image-Edit-2509-prompt-1226-freezevl.sh`

```bash
# 在 DiffSynth-Studio 根目录下执行
bash examples/qwen_image/model_training/lora/Qwen-Image-Edit-2509-prompt-1226-freezevl.sh
```

### 关键训练参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--dataset_base_path` | `/scratch/gym` | 数据根目录 |
| `--dataset_metadata_path` | `...pairs_summary_data_sample_3000_train_prompt_1226.csv` | 训练数据索引 |
| `--data_file_keys` | `image,edit_image` | CSV 中图像列名 |
| `--extra_inputs` | `edit_image` | 额外输入（编辑前图像） |
| `--max_pixels` | `1048576` (1024×1024) | 最大像素数 |
| `--dataset_repeat` | `10` | 数据集重复次数 |
| `--learning_rate` | `1e-4` | 学习率 |
| `--num_epochs` | `100` | 训练轮数 |
| `--lora_base_model` | `dit` | LoRA 应用于 DiT |
| `--lora_rank` | `32` | LoRA 秩 |
| `--lora_target_modules` | `to_q,to_k,to_v,add_q_proj,...` | LoRA 目标模块 |
| `--output_path` | `models/train/Qwen-Image-Edit-2509_data_sample_3000_freezeqwenvl_lora_prompt1226` | 模型保存路径 |

### freeze-VL vs. 普通训练

- **`Qwen-Image-Edit-2509-prompt-1226-freezevl.sh`**（推荐）：使用原始 `text_encoder/model*.safetensors`，VL 编码器冻结不参与训练
- **`Qwen-Image-Edit-2509.sh`**：使用 `checkpoint-225-merged/model*.safetensors`，即经过 SFT 微调的 text encoder

---

## 推理

### DiffSynth Pipeline 推理

```bash
# 在 DiffSynth-Studio 根目录下执行
python examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2509.py
```

推理脚本会：
1. 加载基础模型 + LoRA 权重
2. 遍历数据目录中所有包含 `prev.png` 和对应 prompt 文件的 pair 目录
3. 生成 `next_*_epoch_*_*.png` 保存到对应 pair 目录
4. 支持断点续跑（自动跳过已存在的输出文件）

### 关键推理参数

| 参数 | 值 |
|------|----|
| `seed` | `123` |
| `num_inference_steps` | `40` |
| `height/width` | 与输入图像一致 |
| `torch_dtype` | `torch.bfloat16` |

---

## 环境配置

### 依赖

```bash
pip install -r requirements.txt
```

### 实验追踪

训练使用 **WandB** 进行实验跟踪：
- Project: `diffsynth-qwen-image`
- Run name: `lora-2509`

---

## 快速复现

```bash
# 1. 进入项目目录
cd DiffSynth-Studio

# 2. 创建模型软链接（首次）
bash examples/qwen_image/model_training/lora/soft-link.sh

# 3. 启动训练
bash examples/qwen_image/model_training/lora/Qwen-Image-Edit-2509-prompt-1226-freezevl.sh

# 4. 推理验证（指定 epoch 和数据路径后运行）
python examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2509.py
```
