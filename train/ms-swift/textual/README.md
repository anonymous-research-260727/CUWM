# Textual World Model for Office Applications

本项目基于 **Qwen2.5-VL-7B-Instruct** 模型，通过 LoRA 微调训练一个 **Textual World Model**，用于预测 Office 应用（Word / Excel / PowerPoint）在执行某个 GUI 操作后，下一帧 UI 截图的文本描述。

## 项目结构

```
textual/
├── README.md                        # 本文档
├── process_ms_swift_data.py         # 数据处理：构建 ms-swift 格式训练/测试集
├── vl_sft.sh                        # 训练脚本：LoRA SFT 训练
├── vl_infer.py                      # 推理脚本：批量推理生成预测结果
├── data/                            # 处理后的训练/测试数据 (JSON)
│   ├── train_data_1231_2867.json
│   └── test_data_1231_339.json
├── data_gen/                        # 数据标注：GT 生成脚本
│   └── generate_gt_pred_prompt.py
├── prompts/                         # Prompt 模板
│   ├── textual_wm.py               #   GT 生成 & 预测推理 prompt
│   ├── textual_wm_eval.py          #   LLM-as-Judge 评测 prompt
│   └── gui_descriptions.py         #   各 Office 应用 GUI 操作的描述定义
├── eval/                            # 评测脚本
│   ├── textual_wm_eval.py          #   LLM Judge + ROUGE 评测
│   ├── textual_wm_eval_with_action.py  #  扩展评测（含 Action Coherence）
│   ├── action_evaluation_custom.py #   Action matching evaluator
│   └── cloudgpt_aoai.py            #   OpenAI API 工具
└── utils/                           # 工具函数
    ├── cloudgpt_aoai.py            #   OpenAI API 调用封装
    ├── tool_definitions.py         #   Action 参数标准化定义
    └── ...                         #   其他 API 工具 (Gemini, Qwen, etc.)
```

## 数据与模型位置

### 数据位置

| 数据 | 路径 |
|------|------|
| 原始数据（训练集） | `/path/to/cuwm/data_sample_3000/new_prompt/train` |
| 原始数据（测试集） | `/path/to/cuwm/data_sample_3000/new_prompt/test` |
| 处理后训练数据 | `textual/data/train_data_1231_2867.json` |
| 处理后测试数据 | `textual/data/test_data_1231_339.json` |

### 模型位置

| 模型 | 路径 |
|------|------|
| 基础模型 Qwen2.5-VL-7B-Instruct | `Qwen2.5-VL-7B-Instruct` |
| SFT checkpoint-450 (merged) | `<YOUR_AZURE_BLOB_ROOT>/lora_ckpt/lora_qwenvl_ckpt_yr_prompt_1228/v3-20251231-143109/checkpoint-450-merged/` |
---

## 整体流程

```
数据标注 (GT Generation)
        ↓
数据处理 (Data Processing)
        ↓
模型训练 (LoRA SFT)
        ↓
模型推理 (Inference)
        ↓
评测 (Evaluation)
```

---

## 1. 数据标注 (GT Generation)

### 1.1 数据格式

原始数据以 pair 文件夹组织，每个 pair 包含：

```
pair_xxx/
├── prev.png          # 当前 UI 截图（操作前）
├── next.png          # 下一帧 UI 截图（操作后）
├── action.json       # 用户执行的操作（结构化 JSON）
├── request.txt       # 用户任务指令
├── a11y.json         # Accessibility 信息
└── prompt_nl_gt.txt  # [生成物] Ground Truth 文本描述
```

### 1.2 生成 Ground Truth

使用 GPT 模型（如 `gpt-5.2-chat-20251211`）根据 `prev.png` + `next.png` + `action.json` 生成对下一帧截图的文本描述作为 GT。

**脚本：** `data_gen/generate_gt_pred_prompt.py`

```bash
cd textual
python data_gen/generate_gt_pred_prompt.py
```

**核心逻辑：**
1. 遍历数据目录中所有包含 `prev.png` 的文件夹
2. 根据文件夹路径判断应用类型（Word / Excel / PPT）
3. 读取 `action.json`，匹配对应的 `GUI_DESCRIPTION`
4. 使用 `TEXT_GT_GENERATION_PROMPT` 构造 prompt，将 prev + next 截图编码为 base64 传入 GPT
5. GPT 输出保存为 `prompt_nl_gt.txt`
6. 使用 `ThreadPoolExecutor` 并发处理（默认 15 workers）

**Prompt 模板（`prompts/textual_wm.py` 中的 `TEXT_GT_GENERATION_PROMPT`）要求模型：**
- 说明应用类型
- 简述用户操作
- 以一段连贯文字描述 Next UI Screenshot，按以下区域组织：Title Bar、Ribbon、Main Editing Area、Sidebar/Pane、Navigation Area、Status Bar、Dropdown/Popout

### 1.3 配置项

修改 `data_gen/generate_gt_pred_prompt.py` 末尾的参数：
```python
root = "/path/to/your/data/train"   # 数据根目录
max_workers = 15                      # 并发数
```

---

## 2. 数据处理 (Data Processing)

将标注好的数据转换为 ms-swift 框架要求的 JSON 格式。

**脚本：** `process_ms_swift_data.py`

```bash
cd textual
python process_ms_swift_data.py
```

### 2.1 输入要求

每个 pair 文件夹需包含：
- `prev.png` — 当前 UI 截图
- `next.png` — 下一帧 UI 截图
- `action.json` — 操作 JSON
- `prompt_nl_gt.txt` — GT 文本描述（第 1 步生成）

### 2.2 输出格式

生成的 JSON 文件（保存在 `data/` 目录），每条样本格式如下：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<TEXT_PRED_PREDICTION_PROMPT with app_name, action, gui_description>"
    },
    {
      "role": "assistant",
      "content": "<prompt_nl_gt.txt 的内容>"
    }
  ],
  "images": ["path/to/prev.png"]
}
```

**注意：** 训练时只使用 `prev.png`（模型需要根据当前截图 + action 预测下一帧描述）。

### 2.3 配置项

修改脚本末尾的 `DATA_ROOT`：
```python
# 训练集
DATA_ROOT = "/path/to/data/train"
# 测试集
DATA_ROOT = "/path/to/data/test"
```

---

## 3. 模型训练 (LoRA SFT)

使用 [ms-swift](https://github.com/modelscope/ms-swift) 框架对 Qwen2.5-VL-7B-Instruct 进行 LoRA 微调。

**脚本：** `vl_sft.sh`

```bash
cd textual
bash vl_sft.sh
```

### 3.1 训练配置

| 参数 | 值 |
|------|-----|
| 基础模型 | `Qwen2.5-VL-7B-Instruct` |
| 训练方式 | LoRA |
| LoRA Rank | 32 |
| LoRA Alpha | 32 |
| Target Modules | all-linear |
| 精度 | bfloat16 |
| GPU | 4 卡 (CUDA 0,1,2,3) |
| Batch Size | 4 per device |
| Gradient Accumulation | 4 steps |
| 有效 Batch Size | 64 |
| Learning Rate | 1e-4 |
| Epochs | 10 |
| Max Length | 8192 |
| Warmup Ratio | 0.05 |
| DeepSpeed | ZeRO-2 |
| 保存策略 | 每个 epoch 保存 |
| Logging | wandb |

### 3.2 关键参数说明

```bash
swift sft \
    --model /path/to/Qwen2.5-VL-7B-Instruct \
    --train_type lora \
    --dataset '/path/to/train_data.json' \
    --lora_rank 32 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --output_dir output/prompt_1228 \
    --deepspeed zero2
```

### 3.3 LoRA 合并

训练完成后，需将 LoRA 权重合并回基础模型以便推理：

```bash
swift export \
    --model /path/to/Qwen2.5-VL-7B-Instruct \
    --adapters output/prompt_1228/checkpoint-xxx \
    --merge_lora true \
    --output_dir output/prompt_1228/checkpoint-xxx-merged
```

---

## 4. 模型推理 (Inference)

使用训练好的模型（或基线模型）对测试集进行批量推理。

**脚本：** `vl_infer.py`

```bash
cd textual
python vl_infer.py
```

### 4.1 配置

修改 `vl_infer.py` 中的关键参数：

```python
model_key = "sft-ckpt-45"   # 选择模型（见 model_dict）
infer_backend = "vllm"       # 推理后端: "vllm" 或 "pt"
```

支持的模型包括：
- `base` — 原始 Qwen2.5-VL-7B-Instruct
- `sft-ckpt-45` — SFT checkpoint-45 (合并后)
- `sft-ckpt-450` — SFT checkpoint-450 (合并后)
- `checkpoint-450-grpo` — GRPO 训练 checkpoint
- `gpt-5.2-chat-20251211` — GPT baseline（通过 API 调用）

### 4.2 输出

推理结果保存至 `output/{model_key}_{infer_backend}.json`，在原始数据基础上新增：
- `gt` — Ground Truth 文本描述
- `pred` — 模型预测的文本描述

---

## 5. 评测 (Evaluation)

### 5.1 基础评测：LLM-as-Judge + ROUGE

**脚本：** `eval/textual_wm_eval.py`

```bash
cd textual/eval
python textual_wm_eval.py
```

#### 评测指标

**LLM-as-Judge（GPT 评分）**：对每条样本的 GT 与 PRED 进行多维度对比打分：

| 维度 | 权重 | 说明 |
|------|------|------|
| App Name | 0.10 | 应用类型是否正确 |
| User Action | 0.175 | 用户操作描述是否准确 |
| Title Bar | 0.125 | 标题栏变化 |
| Ribbon | 0.1375 | 功能区变化 |
| Main Editing Area | 0.1875 | 主编辑区域变化（权重最高） |
| Sidebar / Pane | 0.10 | 侧边栏变化 |
| Navigation Area | 0.075 | 导航区域变化 |
| Status Bar | 0.10 | 状态栏变化 |

每个维度评分为 0 / 0.5 / 1（完全错误 / 部分正确 / 完全正确）。

**ROUGE 指标**：计算 ROUGE-1、ROUGE-2、ROUGE-L F1 分数。

#### 输出文件

- `output/{model_key}-llm-eval-in-gpt-5.2-20251211.json` — 每条样本的详细评分
- `output/{model_key}-eval-summary.json` — 汇总结果（各维度平均分 + 加权平均 + ROUGE）

### 5.2 扩展评测：Action Coherence

**脚本：** `eval/textual_wm_eval_with_action.py`

```bash
cd textual/eval
python textual_wm_eval_with_action.py
```

在基础评测之上，增加 **Action Coherence** 评测，衡量文本描述是否能帮助下游 Agent 正确预测动作：

1. 用模型的 PRED 描述 + a11y 信息，让 Action Model 预测下一步操作 → `action_pred`
2. 用 GT 描述 + 真实截图 + a11y 信息，让同一 Action Model 预测操作 → `action_gt`
3. 比较 `action_pred` 与 `action_gt` 的匹配度

#### Action 匹配评分

| 匹配项 | 分值 |
|--------|------|
| Function Match（操作类型一致） | 0.25 |
| Status Match（状态一致） | 0.25 |
| Args Match（参数一致） | 0.50 |
| **总分** | **1.00** |

参数匹配支持：
- `control_label` 精确匹配
- `coordinate` 容错匹配（tolerance = 25px）
- `drag` 起止点匹配

#### 输出文件

- `output_action/{model_key}_{action_model}_action-eval.json` — 每条样本的 action 评测详情
- `output_action/{model_key}_{action_model}_action-summary.json` — Action Coherence 汇总

---

## 快速开始

```bash
# 1. 数据标注（生成 GT）
python data_gen/generate_gt_pred_prompt.py

# 2. 数据处理（生成训练/测试 JSON）
python process_ms_swift_data.py

# 3. 训练
bash vl_sft.sh

# 4. LoRA 合并
swift export --model /path/to/base_model \
    --adapters output/prompt_1228/checkpoint-xxx \
    --merge_lora true \
    --output_dir output/prompt_1228/checkpoint-xxx-merged

# 5. 推理
python vl_infer.py

# 6. 评测
cd eval && python textual_wm_eval.py
# 或扩展评测
cd eval && python textual_wm_eval_with_action.py
```

## 依赖

- Python 3.10+
- [ms-swift](https://github.com/modelscope/ms-swift)
- vLLM（推理加速）
- rouge
- openai
- torch
- tqdm
