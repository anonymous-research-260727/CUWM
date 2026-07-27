# UI World Model — GRPO RL Training for Office GUI Agent

This project uses **GRPO (Group Relative Policy Optimization)** to perform reinforcement learning fine-tuning on a **Qwen2.5-VL-7B** vision-language model, enabling it to serve as a *world model* for Office GUI automation (Word, Excel, PowerPoint). Given a screenshot and a task instruction, the model predicts a detailed textual description of the next UI state after executing an action.

## Directory Structure

```
ui_world_model/
├── build_dataset_ui_world_model.py          # Build train/test parquet datasets from SFT JSON
├── build_dataset_ui_world_model_delete_last_pair.py  # Variant: remove last action-pair
├── reward_ui_world_model_action.py          # Main reward function (action match + GPT judge + length penalty)
├── reward_ui_world_model_base.py            # Base reward function (GPT/Qwen judge only)
├── reward_ui_world_model_judge_len.py       # Reward with Qwen judge + length penalty
├── reward_ui_world_model_pairwise.py        # Pairwise reward (embedding similarity + ROUGE + GPT judge)
├── reward_yr.py                             # Additional reward variant
├── action_evaluation.py                     # Action command matching evaluation (function / control / args)
├── tool_definitions.py                      # Office action tool schemas (click, type, drag, etc.)
├── prompts.py                               # Prompt templates for GUI action selection
├── embedding.py                             # Embedding client (OpenAI / local sentence-transformers)
├── gpt_api.py                               # GPT-based pairwise judge API wrapper
├── qwen_api.py                              # Qwen-based judge API wrapper
├── qwenvl_api.py                            # QwenVL multimodal judge API wrapper
├── cloudgpt_aoai.py                         # Azure OpenAI client helper
├── run_qwen2_5_vl-7b-officewm.sh           # Main training launch script
├── run_qwen2_5_vl-7b-officewm_judge_len.sh # Training script (judge + length penalty variant)
├── read.ipynb                               # Exploration / analysis notebook
├── data/                                    # Parquet datasets (train / test)
│   ├── train.parquet
│   ├── train_1k.parquet
│   ├── test.parquet
│   └── test_mini.parquet
└── no_last/                                 # Datasets with last action-pair removed
```

## Quick Start

### 1. Prepare Dataset

Convert SFT JSON data into Parquet format consumable by verl:

```bash
python build_dataset_ui_world_model.py \
    --sft_json /path/to/train_data.json \
    --local_save_dir /path/to/output \
    --split train
```

Each sample contains:
- **prompt**: user message with task instruction & screenshot
- **images**: the current (prev) screenshot as a PIL Image
- **reward_model.ground_truth**: reference assistant response (next-state description)
- **extra_info**: metadata including ground-truth action JSON, file paths to `prev.png`, `next.png`, `a11y.json`, etc.

### 2. Launch GRPO Training

```bash
bash run_qwen2_5_vl-7b-officewm.sh [vllm]
```

Key training configurations:
| Parameter | Value |
|---|---|
| Algorithm | GRPO |
| Base model | Qwen2.5-VL-7B (SFT checkpoint) |
| LoRA rank / alpha | 64 / 32 |
| Learning rate | 3e-6 |
| Rollout samples (n) | 5 |
| GPUs | 2 |
| Max prompt length | 4096 tokens |
| Max response length | 2048 tokens |
| Train batch size | 32 |

## Reward Design

The reward function (`compute_score`) in `reward_ui_world_model_action.py` combines multiple signals:

### Base Reward — GPT Judge (weighted)
A GPT-based judge scores the predicted next-state description against the ground-truth across 8 UI regions with per-region weights:

| Region | Weight |
|---|---|
| `main_editing_area` | 1.5 |
| `user_action` | 1.4 |
| `ribbon` | 1.1 |
| `title_bar` | 1.0 |
| `app_name` | 0.8 |
| `sidebar_pane` | 0.8 |
| `status_bar` | 0.8 |
| `navigation_area` | 0.6 |

### Length Penalty (asymmetric, token-based)
- **Too long** (ratio > 1.25): severe penalty up to 0.45
- **Too short** (ratio < 0.75): mild penalty up to 0.25

### Action Match Reward
Uses `ActionEvaluationA11y` to compare the predicted action (parsed from the model's output via an auxiliary GPT call) against the ground-truth action. Evaluates:
- **function match**: action type (click, type, drag, etc.)
- **control_name match**: target UI control
- **args match**: coordinates, text content, etc.

Final reward:
```
reward = (1 - action_mix) * (base_reward - len_penalty) + action_mix * action_match
```
Clamped to `[0, 1]`.

### Reward Variants

| File | Description |
|---|---|
| `reward_ui_world_model_base.py` | GPT/Qwen judge only |
| `reward_ui_world_model_judge_len.py` | Qwen judge + length penalty |
| `reward_ui_world_model_pairwise.py` | GPT judge + embedding similarity + ROUGE |
| `reward_ui_world_model_action.py` | Full reward (judge + length penalty + action match) |

## End-to-End Example

Below is a walkthrough illustrating the full pipeline — from raw data to training — using a single PowerPoint sample.

### Step 1: Raw Data Layout

Each training sample lives in a *pair directory* with the following structure:

```
excel/bing_search/paired/excel_4_1454/pair_01/
├── prev.png          # Screenshot BEFORE the action
├── next.png          # Screenshot AFTER the action
├── action.json       # Ground-truth action command
├── a11y.json         # Accessibility tree of the current UI
├── request.txt       # Task instruction in natural language
└── prompt_nl_gt.txt  # Ground-truth next-state description (for reward)
```

**action.json** example:
```json
{
  "function": "click",
  "args": { "control_label": 15, "coordinate": null, "button": "left" },
  "control_name": "Insert",
  "status": "CONTINUE"
}
```

**request.txt** example:
```
Insert a Microsoft Forms quiz into the current slide.
```

### Step 2: SFT JSON → Parquet

The SFT JSON (produced by an upstream annotation pipeline) looks like:

```json
[
  {
    "images": ["/path/to/pair_01/prev.png"],
    "messages": [
      {
        "role": "user",
        "content": "<image>\nYou are a World Model ... Task: Insert a Microsoft Forms quiz ..."
      },
      {
        "role": "assistant",
        "content": "This is Microsoft PowerPoint. The user clicked on 'Insert' tab ..."
      }
    ]
  }
]
```

Convert it to a Parquet dataset:

```bash
python build_dataset_ui_world_model.py \
    --sft_json /path/to/train_data_2867.json \
    --local_save_dir ./data \
    --split train
```

The resulting Parquet contains columns:

| Column | Content |
|---|---|
| `data_source` | `"ui_world_model_rl"` |
| `prompt` | User message (with `<image>` placeholder) |
| `images` | PIL Image of `prev.png` |
| `reward_model.ground_truth` | Reference assistant text |
| `extra_info.groundtruth_action` | Action JSON string |
| `extra_info.pair_path` | Path to pair directory (for reward file lookup) |
| `extra_info.prev_path` / `next_path` | Absolute paths to screenshots |

### Step 3: GRPO Training

Launch training with 2 GPUs:

```bash
bash run_qwen2_5_vl-7b-officewm.sh vllm
```

During each rollout iteration, the model generates `n=5` candidate next-state descriptions for each prompt. The reward function scores every candidate:

### Step 4: Reward Scoring Example

```python
from examples.ui_world_model.reward_ui_world_model_action import compute_score

pred = (
    'This is Microsoft PowerPoint. The user has clicked on the "Insert" tab '
    'in the Ribbon, which is now active. The Main Editing Area has been updated '
    'to display a new slide with the text "Quarterly Report" centered. '
    'The sidebar has been updated to include a new panel labeled "Design Ideas".'
)

gt = (
    'This is Microsoft PowerPoint. The user interaction was a single click on '
    'a control within the inserted Microsoft Forms object, which transitioned '
    'the embedded content from a selection screen to a different internal state. '
    'The key change is in the Main Editing Area: the previously visible embedded '
    'Forms selection interface has been replaced by a simplified Microsoft Forms '
    'placeholder view ...'
)

result = compute_score(
    data_source="ui_world_model_rl",
    solution_str=pred,
    ground_truth=gt,
    extra_info={
        "groundtruth_action": '{"function": "click"}',
        "pair_path": "/path/to/excel_4_1454/pair_01",
    },
    use_length_penalty=True,
)
```

**Output** (logged to wandb):

```python
{
    "score": 0.42,                    # Final blended reward ∈ [0, 1]
    "reward/base": 0.55,              # GPT judge weighted score
    "reward/len_penalty": 0.08,       # Token-length penalty (pred was shorter)
    "reward/length_ratio": 0.61,      # pred_tokens / gt_tokens
    "reward/pred_tokens": 98,
    "reward/gt_tokens": 160,
    "reward/len_penalty_type": "short_mild",
    "reward/action_match": 0.25,      # Action coherence (function matched, args didn't)
    "reward/use_action_reward": 1.0,
    "reward/use_len_penalty": 1.0,
}
```

**How the final score is computed:**

```
base_after_penalty = base (0.55) − len_penalty (0.08) = 0.47
action_mix = 0.2  (default)

score = (1 − 0.2) × 0.47 + 0.2 × 0.25
      = 0.376 + 0.05
      = 0.426  →  clamp to [0, 1]  →  0.42
```

### Step 5: Monitor with Wandb

The training script logs to project `verl_grpo_officeWM3k_action_next`. Key metrics to watch:

- `reward/score` — overall reward trend (should increase)
- `reward/base` — GPT judge quality (should increase)
- `reward/len_penalty` — length penalty (should decrease as model calibrates output length)
- `reward/action_match` — action coherence (should increase)
- `actor/entropy` — policy entropy (gradual decrease is healthy)

## Supported Actions

Defined in `tool_definitions.py`, supporting both coordinate-based and accessibility-label-based control:

- `click` — single/double click at coordinate or control label
- `type` — type text at a location
- `drag` — drag from start to end coordinate
- `wheel_mouse_input` — scroll wheel
- `insert_table`, `select_text`, `select_table`, `select_paragraph` — Word-specific
- `insert_excel_table`, `select_table_range`, `set_cell_value`, `auto_fill`, `reorder_columns` — Excel-specific
- `set_background_color` — PowerPoint-specific
- `save_as`, `set_font` — cross-app utilities

## Data & Model Locations

### Raw SFT Data (Source JSON)

| File | Path | Description |
|---|---|---|
| train (2867 samples) | `<YOUR_AZURE_BLOB_ROOT>/project/rl-0.6.0/verl/examples/ui_world_model/sft_data/train_data_1231_2867.json` | Full training set |
| train (991 samples) | `<YOUR_AZURE_BLOB_ROOT>/project/rl-0.6.0/verl/examples/ui_world_model/sft_data/train_data_0131_991.json` | new 1k training     |
| test (339 samples) | `<YOUR_AZURE_BLOB_ROOT>/project/rl-0.6.0/verl/examples/ui_world_model/sft_data/test_data_1231_339.json` | Test / evaluation set |

### Processed Parquet Datasets

Used directly by the training scripts.

| Dataset | Path |
|---|---|
| train (full) | `.../ui_world_model/data/train.parquet` |
| train (1k new) | `.../ui_world_model/data/train_1k.parquet` |
| test | `.../ui_world_model/data/test.parquet` |
| test (mini) | `.../ui_world_model/data/test_mini.parquet` |
| train (no last pair) | `.../ui_world_model/no_last/train.parquet` |
| test mini (no last pair) | `.../ui_world_model/no_last/test_mini_no_last.parquet` |

> **Note:** `no_last/` variants are built by `build_dataset_ui_world_model_delete_last_pair.py`, which removes the final action-pair from each trajectory to avoid data leakage.

### Raw Image-Pair Data

The pair directories referenced by `extra_info.pair_path` are located under:

```
/path/to/cuwm/data_sample_3000/new_prompt/train_for_grpo/
├── excel/
│   └── bing_search/paired/excel_*/pair_*/
├── word/
│   └── .../paired/word_*/pair_*/
└── ppt/
    └── .../paired/ppt_*/pair_*/
```

Each `pair_XX/` directory contains: `prev.png`, `next.png`, `action.json`, `a11y.json`, `request.txt`, `prompt_nl_gt.txt`.

### Model Checkpoints

| Model | Path | Description |
|---|---|---|
| SFT base (Qwen2.5-VL-7B + LoRA, merged) | `<YOUR_AZURE_BLOB_ROOT>/project/rl-0.6.0/verl/examples/ui_world_model/model/lora_qwenvl_ckpt_yr_prompt_1228/v3-20251231-143109/checkpoint-450-merged` | Used as the initial policy for GRPO training |
| SFT intermediate ckpts | `.../v3-20251231-143109/checkpoint-{45,90,...,450}/` | LoRA-only checkpoints (not merged) for analysis |

> RL training checkpoints are saved by verl to `<YOUR_AZURE_BLOB_ROOT>/project/rl-0.6.0/verl/checkpoints/verl_grpo_officeWM3k/qwen2_5_vl_7b_officeWM/global_step_250/actor/huggingface` (defaults to `checkpoints/` under the working directory), at a frequency of every 50 steps (`trainer.save_freq=50`).

## Dependencies

- [verl](https://github.com/volcengine/verl) (v0.6.0) — RL training framework
- `vllm` — rollout engine
- `transformers`, `datasets` — Hugging Face ecosystem
- `openai` — GPT judge API calls
- `tiktoken` — token counting for length penalty
- `sentence-transformers` — local embedding similarity (optional)
- `Pillow` — image handling
