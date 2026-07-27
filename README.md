# CUWM: World Model Evaluation for Office GUI Agents

A framework for training and evaluating world models that predict UI outcomes in Microsoft Office (Word, Excel, PowerPoint). World models forecast what the screen will look like after an action, helping GUI agents pick better actions.

## Pipeline

```
Current Screenshot + Task Instruction
            │
            ▼
┌───────────────────────────┐
│ 1. Option Generation      │  LLM generates N candidate actions
└─────────┬─────────────────┘
          ▼
┌───────────────────────────┐
│ 2. Outcome Synthesis      │  World model predicts result of each action:
│   • Text: VLM describes   │    - Textual WM (fine-tuned VLM)
│     UI changes            │    - Image WM (diffusion model)
│   • Image: diffusion      │
│     generates next frame  │
└─────────┬─────────────────┘
          ▼
┌───────────────────────────┐
│ 3. Action Selection       │  LLM picks the best action using predictions
└─────────┬─────────────────┘
          ▼
┌───────────────────────────┐
│ 4. Evaluation             │  Compare to ground truth
└───────────────────────────┘
```

**Evaluation modes:** `none` (baseline), `text`, `image`, `text+image`

## Setup

```bash
# Python 3.10+
pip install -e .

# Copy and fill in API keys
cp .env.example .env
```

Required API keys (depending on which models you use):
- `QWEN_API_KEY` / `DASHSCOPE_API_KEY` — Qwen VLM via DashScope
- `GEMINI_API_KEY` — Google Gemini
- `WANDB_API_KEY` — Weights & Biases logging

## Repository Structure

```
├── eval/                   # Core evaluation pipeline
│   ├── agent_wm_gen.py     # Main orchestrator (option gen → synthesis → selection)
│   ├── agent_wm_eval.py    # Accuracy metrics
│   ├── action_evaluation.py# Action comparison logic
│   ├── textual_wm_eval.py  # ROUGE + LLM-as-judge for text WM
│   ├── eval_image_metrics.py  # Image metrics (PSNR, SSIM, LPIPS, FID)
│   └── prompts.py          # Prompt templates for action selection
├── textual/                # Textual world model
│   ├── vl_infer.py         # VLM inference (base or fine-tuned)
│   └── process_ms_swift_data.py  # Data format conversion
├── infer/                  # Image generation APIs
│   ├── Qwen-Image-Edit-2509-api.py
│   └── GPT5-Image-Infer.py
├── train/                  # Training frameworks (vendored)
│   ├── ms-swift/           # VLM fine-tuning (LoRA SFT)
│   ├── DiffSynth-Studio/   # Diffusion model training
│   └── verl/               # RL training (GRPO)
├── utils/                  # LLM API wrappers (Qwen, Gemini, OpenAI-compatible APIs)
├── prompts/                # Prompt templates for world model tasks
├── config/                 # Model path configuration
├── scripts/                # Helper scripts for data processing & eval runs
└── data/                   # Train/test datasets (MS-Swift format)
```

## Usage

### Agent Evaluation (main pipeline)

```bash
# Baseline (no world model)
python -m eval.agent_wm_gen \
  --exp_name baseline \
  --mode none \
  --data_folder new_prompt/test

# With textual world model
python -m eval.agent_wm_gen \
  --exp_name text_wm \
  --mode text \
  --data_folder new_prompt/test

# With image world model
python -m eval.agent_wm_gen \
  --exp_name image_wm \
  --mode image \
  --model_key epoch-24 \
  --data_folder new_prompt/test

# With both
python -m eval.agent_wm_gen \
  --exp_name text_image_wm \
  --mode text+image \
  --model_key epoch-24 \
  --data_folder new_prompt/test
```

Key arguments:
| Argument | Description |
|----------|-------------|
| `--exp_name` | Experiment name (output subdirectory) |
| `--mode` | `none`, `text`, `image`, or `text+image` |
| `--model_key` | Image WM model: `base`, `epoch-24`, `gpt-image-1.5` |
| `--num_options` | Number of candidate actions (default: 5) |
| `--prompt_version` | Prompt variant: `v1`, `v2`, `v3` |
| `--parallel` | Number of parallel workers |
| `--use_reasoning_prompt` | Enable chain-of-thought reasoning |
| `--qwen_lora_path` | Path to LoRA adapter for local VLM |

### Compute Evaluation Metrics

```bash
# Action accuracy
python -m eval.agent_wm_eval --exp_name <experiment_name>

# Image quality (PSNR, SSIM, LPIPS, FID)
python -m eval.eval_image_metrics --exp_name <experiment_name>

# Textual WM quality (ROUGE + LLM judge)
python -m eval.textual_wm_eval --exp_name <experiment_name>
```

### Training

**Textual WM** (LoRA fine-tune Qwen2.5-VL-7B):
```bash
# 1. Prepare data
python -m textual.process_ms_swift_data

# 2. Fine-tune
bash textual/vl_sft.sh
```

**Image WM** (LoRA fine-tune Qwen-Image-Edit-2509):
```bash
cd train/DiffSynth-Studio
bash examples/qwen_image/model_training/lora/Qwen-Image-Edit-2509-prompt-1226-freezevl.sh
```

**RL Training** (GRPO on Qwen2.5-VL-7B):
```bash
cd train/verl
bash examples/ui_world_model/run_qwen2_5_vl-7b-officewm.sh
```

## Data Format

Each sample is a paired UI transition:

```
pair_XX/
├── prev.png              # Current screenshot
├── next.png              # Ground-truth next screenshot
├── prev_annotated.png    # Screenshot with numbered UI element markers
├── action.json           # Ground-truth action (function, args, control)
├── a11y.json             # Accessibility tree
├── request.txt           # User task instruction
└── control_info.json     # UI control labels
```

Training data is provided in MS-Swift JSON format in `data/`.

## Evaluation Metrics

**Agent evaluation:** function match, args match, overall action match accuracy

**Image WM quality:** PSNR, SSIM, LPIPS, FID

**Textual WM quality:** ROUGE-1/2/L, LLM-as-a-Judge

## Supported Models

| Role | Models |
|------|--------|
| Action generation / selection | GPT-4o, GPT-5.2, Gemini, Qwen3-VL-8B |
| Textual world model | Qwen2.5-VL-7B (base or LoRA fine-tuned) |
| Image world model | Qwen-Image-Edit-2509 (base or LoRA fine-tuned), GPT-Image |

## License

MIT
