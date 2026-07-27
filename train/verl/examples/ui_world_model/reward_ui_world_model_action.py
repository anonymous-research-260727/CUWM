import os
import json
import time
import hashlib
import re
import base64
import glob
from typing import Any, Dict, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from examples.ui_world_model.gpt_api import GPTJudge
from examples.ui_world_model.action_evaluation import ActionEvaluationA11y
import random

# -----------------------------
# Globals
# -----------------------------
actEval = ActionEvaluationA11y()

DEFAULT_WEIGHTS = {
    "app_name": 0.8,
    "user_action": 1.4,
    "title_bar": 1.0,
    "ribbon": 1.1,
    "main_editing_area": 1.5,
    "sidebar_pane": 0.8,
    "navigation_area": 0.6,
    "status_bar": 0.8,
}


# -----------------------------
# Base reward helpers
# -----------------------------
def _weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    num = 0.0
    den = 0.0
    for k, w in weights.items():
        den += float(w)
        num += float(w) * float(scores.get(k, 0.0))
    if den <= 0:
        return 0.0
    return num / den


def _count_tokens_tiktoken(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens using tiktoken. Works well as a proxy length measure.
    """
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough estimate if tiktoken unavailable
        return max(1, len(text) // 4)


def _simple_asym_length_penalty_by_tokens(
    pred: str,
    gt: str,
    lower_ratio: float = 0.75,
    upper_ratio: float = 1.25,
    severe_max_penalty: float = 0.45,
    mild_max_penalty: float = 0.25,
    severe_alpha: float = 1.0,
    no_short_penalty: bool = False,
    encoding_name: str = "cl100k_base",
) -> Tuple[float, Optional[int], Optional[int], Optional[float], str]:
    """
    简单非对称长度惩罚（token 比例）：
    - pred/gt > upper_ratio: 严厉惩罚（cap 到 severe_max_penalty）
    - pred/gt < lower_ratio: 轻微惩罚（或 no_short_penalty=True 则不惩罚）
    - 其他：不惩罚

    返回: (penalty, pred_tokens, gt_tokens, ratio, penalty_type)
    """
    gt_tokens = _count_tokens_tiktoken(gt, encoding_name=encoding_name)
    pred_tokens = _count_tokens_tiktoken(pred, encoding_name=encoding_name)

    if gt_tokens <= 0:
        return 0.0, pred_tokens, gt_tokens, None, "none"

    ratio = pred_tokens / float(max(gt_tokens, 1))

    # 区间内不惩罚
    if lower_ratio <= ratio <= upper_ratio:
        return 0.0, pred_tokens, gt_tokens, ratio, "none"

    # 过短：不严厉（轻微 or 不惩罚）
    if ratio < lower_ratio:
        if no_short_penalty:
            return 0.0, pred_tokens, gt_tokens, ratio, "short_none"
        gap = (lower_ratio - ratio) / max(lower_ratio, 1e-6)
        pen = mild_max_penalty * min(1.0, max(0.0, gap))
        return float(pen), pred_tokens, gt_tokens, ratio, "short_mild"

    # 超长：严厉惩罚
    gap = (ratio - upper_ratio) / max(upper_ratio, 1e-6)
    shaped = min(1.0, max(0.0, gap)) ** float(severe_alpha)
    pen = severe_max_penalty * shaped
    return float(pen), pred_tokens, gt_tokens, ratio, "long_severe"


# -----------------------------
# Action coherence helpers (minimal intrusive)
# -----------------------------
try:
    from eval.prompts import SUPPORTED_ACTIONS
except Exception:
    try:
        from examples.ui_world_model.prompts import SUPPORTED_ACTIONS
    except Exception:
        SUPPORTED_ACTIONS = None


ACTION_PREDICTION_A11Y_SYS_PROMPT_GPT = """
You are an expert in Office Application automation and graphical user interfaces with accessibility support.

You will be provided with the following inputs:

1. **Current screenshot (optional)**: An image of the current state of an Office Application. This screenshot may be missing.
2. **Screenshot Description**: A textual description of the current UI state derived from the screenshot.
3. **Accessibility (a11y) information**: This includes a list of control element labels and the textual name of the currently active Office Application.
4. **Task instruction**: A description of the action or goal to be completed.
5. **Supported actions**: A list of all actions that can be performed in this environment.

The accessibility information contains control labels that correspond to UI control elements in the current application state, allowing you to locate and reference specific interface components.

Your objective is to generate the **single best next action** to accomplish the given task instruction, based on the available information, including the screenshot description, accessibility information, task instruction, supported actions, and the current screenshot if it is provided.

Use all the provided information to determine the most appropriate next action. If the current screenshot is not available, rely on the screenshot description and accessibility information.

**IMPORTANT: When possible, prioritize using control_label over coordinate for actions. Control labels are more reliable than raw screen coordinates.**

You must output the next action in JSON format as a JSON array containing **exactly one element**.

Each element must contain only a "tool_call" field.

The "tool_call" field must contain:
- "function": str, The function/action type to execute
- "args": Dict, The arguments/parameters for the function
- "status": str, The status after performing this action (either "CONTINUE" or "FINISH")

Only **ONE** action should be generated.

For example, for click operations, prioritize control_label over coordinate:
```json
{
  "tool_call": {
    "function": "click",
    "args": {"control_label": 15, "coordinate": null, "button": "left"},
    "status": "CONTINUE"
  }
}
````

For example, if control_label is not available, fall back to coordinate:

```json
{
  "tool_call": {
    "function": "click",
    "args": {"control_label": null, "coordinate": [150, 30], "button": "left"},
    "status": "CONTINUE"
  }
}
```

For example, for type operations:

```json
{
  "tool_call": {
    "function": "type",
    "args": {"control_label": 8, "coordinate": null, "keys": "Hello World", "clear_current_text": true},
    "status": "CONTINUE"
  }
}
```

For example, for drag operations:

```json
{
  "tool_call": {
    "function": "drag",
    "args": {"start_coordinate": [100, 100], "end_coordinate": [200, 200], "button": "left"},
    "status": "CONTINUE"
  }
}
```

For example, if the task is already completed, output:

```json
{
  "tool_call": {
    "function": "",
    "args": {},
    "status": "FINISH"
  }
}
```

Your response MUST be a valid JSON array with exactly one element and no additional text.
"""


ACTION_PREDICTION_A11Y_USER_PROMPT_GPT = """
Task instruction:
{instruction}

Screenshot Description:
{screen_description}

Accessibility Information:
{a11y}

Supported actions:
{actions}

The current screenshot may be provided as an image, but it may also be missing.

Please analyze the current state using the available information and output the **single best next action** to move toward completing the task instruction.

Output the result in JSON array format (with exactly one element) and no additional text.
""".strip()


def _encode_image_to_data_url(img_path: str) -> str:
    if not img_path:
        raise ValueError("img_path is empty")
    ext = os.path.splitext(img_path)[1].lower().lstrip(".")
    mime = "image/png" if ext in ("png",) else ("image/jpeg" if ext in ("jpg", "jpeg") else "image/png")
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _infer_app_type_from_path(p: str) -> str:
    lp = (p or "").lower()
    if "ppt" in lp or "powerpoint" in lp:
        return "ppt"
    if "word" in lp:
        return "word"
    if "excel" in lp:
        return "excel"
    raise ValueError(f"Cannot infer app type from path: {p}")


def _call_action_llm(messages, model: str, temperature: float = 0.0, **kwargs) -> str:
    """
    OpenAI 兼容接口调用。
    支持 kwargs 覆盖：
      - action_base_url / action_api_key / action_client
    """
    if model == "qwen3-vl-8b-instruct":
        base_url = "https://api.vectorengine.ai/v1"
        api_key = "<YOUR_API_KEY>"
    if model == "qwen3-vl-flash":
        base_url = "https://api.vectorengine.ai/v1"
        api_key = "<YOUR_API_KEY>"
    client = kwargs.get("action_client")
    if client is None:
        client = OpenAI(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _generate_action_prompt(
    screen_description: Optional[str],
    screen_description_path: Optional[str],
    a11y_path: str,
    instruction_path: str,
) -> str:
    assert screen_description or screen_description_path
    assert not (screen_description and screen_description_path)

    if screen_description_path:
        with open(screen_description_path, "r", encoding="utf-8") as f:
            screen_description = f.read().strip()

    with open(a11y_path, "r", encoding="utf-8") as f:
        a11y = json.load(f)

    with open(instruction_path, "r", encoding="utf-8") as f:
        instruction = f.read().strip()

    app = _infer_app_type_from_path(a11y_path)
    supported_actions = []
    if SUPPORTED_ACTIONS is not None:
        supported_actions = SUPPORTED_ACTIONS.get(app, [])

    usr_prompt = ACTION_PREDICTION_A11Y_USER_PROMPT_GPT.format(
        instruction=instruction,
        screen_description=screen_description,
        a11y=json.dumps(a11y, indent=2),
        actions=supported_actions,
    )
    return ACTION_PREDICTION_A11Y_SYS_PROMPT_GPT + "\n\n" + usr_prompt


def _get_action(
    img_path: Optional[str],
    screen_description: Optional[str],
    screen_description_path: Optional[str],
    a11y_path: str,
    instruction_path: str,
    **kwargs,
):

    prompt = _generate_action_prompt(
        screen_description=screen_description,
        screen_description_path=screen_description_path,
        a11y_path=a11y_path,
        instruction_path=instruction_path,
    )

    content = [{"type": "text", "text": prompt}]
    if img_path:
        content.append({"type": "image_url", "image_url": {"url": _encode_image_to_data_url(img_path)}})

    messages = [{"role": "user", "content": content}]

    action_model = kwargs.get("action_model", "qwen3-vl-flash")  # 改成你实际可用的模型名
    max_retry = int(kwargs.get("action_max_retry", 3))

    for _ in range(max_retry):
        try:
            response = _call_action_llm(messages, model=action_model, temperature=0.0, **kwargs)
            text = response.replace("```json", "").replace("```", "").strip()
            
            try:
                return json.loads(text)["tool_call"]
            except json.JSONDecodeError:
                pass
        except Exception:
            continue
    return None


def _action_coherence_reward(
    textual_wm_response: str, # pair_{i}/next.png description
    img_path: str, # next.png
    gt_description_path: str, # prompt_nl_gt.txt
    a11y_path: str, # pair_{i+1}/a11y.json
    instruction_path: str,
    **kwargs,
) -> float:
    """
    返回 0~1：
      function_match 0.25
      status_match   0.25
      args_match     0.5
    """
    action_pred = _get_action(
        img_path=None,
        screen_description=textual_wm_response,
        screen_description_path=None,
        a11y_path=a11y_path,
        instruction_path=instruction_path,
        **kwargs,
    )
    action_gt = _get_action(
        img_path=img_path,
        screen_description=None,
        screen_description_path=gt_description_path,
        a11y_path=a11y_path,
        instruction_path=instruction_path,
        **kwargs,
    )

    if action_pred is None or action_gt is None:
        return 0.0

    eval_result = actEval.compare_action_command_2_pred(
        gt_raw=None,
        gt_command=action_gt,
        pred_command=action_pred,
    )

    function_match = bool(eval_result.get("function_match", False))
    status_match = bool(eval_result.get("status_match", False))
    args_match = bool(eval_result.get("args_match", False))

    score = 0.0
    if function_match:
        score += 0.25
    if status_match:
        score += 0.25
    if args_match:
        score += 0.5
    return float(score)


def _infer_pair_files(pair_path: str) -> Dict[str, str]:
    """
    从 pair 目录里自动找：
      - next.png / next.*
      - a11y.json：优先取“下一个 pair”的 a11y（如 pair_01 -> pair_02），若不存在则取当前 pair 的
      - request.txt / *request*.txt
      - prompt_nl_gt.txt（优先）/ gt*.txt
    """
    pair_path = pair_path or "."

    # ---------- prev ----------
    # prev = os.path.join(pair_path, "prev.png")
    # if not os.path.exists(prev):
    #     cands = glob.glob(os.path.join(pair_path, "prev.*"))
    #     prev = cands[0] if cands else prev

    # ---------- next ----------
    nxt = os.path.join(pair_path, "next.png")
    if not os.path.exists(nxt):
        cands = glob.glob(os.path.join(pair_path, "next.*"))
        nxt = cands[0] if cands else nxt

    # ---------- a11y (prefer next pair's a11y) ----------
    def _next_pair_dir(cur_pair_dir: str) -> Optional[str]:
        """
        如果 cur_pair_dir 形如 .../pair_01，返回 .../pair_02（同级目录），否则返回 None
        """
        base = os.path.basename(os.path.normpath(cur_pair_dir))
        m = re.match(r"^(pair_)(\d+)$", base)
        if not m:
            return None
        prefix, num = m.group(1), m.group(2)
        width = len(num)
        nxt_num = str(int(num) + 1).zfill(width)
        return os.path.join(os.path.dirname(os.path.normpath(cur_pair_dir)), f"{prefix}{nxt_num}")

    a11y = None

    # 1) try next pair's a11y.json
    npair = _next_pair_dir(pair_path)
    if npair and os.path.isdir(npair):
        cand = os.path.join(npair, "a11y.json")
        if os.path.exists(cand):
            a11y = cand
        else:
            cands = glob.glob(os.path.join(npair, "*a11y*.json"))
            if cands:
                a11y = cands[0]

    # 2) fallback to current pair's a11y.json
    if a11y is None:
        cand = os.path.join(pair_path, "a11y.json")
        if os.path.exists(cand):
            a11y = cand
        else:
            cands = glob.glob(os.path.join(pair_path, "*a11y*.json"))
            a11y = cands[0] if cands else cand

    # ---------- request ----------
    req = os.path.join(pair_path, "request.txt")
    if not os.path.exists(req):
        cands = glob.glob(os.path.join(pair_path, "*request*.txt"))
        req = cands[0] if cands else req

    # ---------- gt description ----------
    gt_cands = glob.glob(os.path.join(pair_path, "prompt_nl_gt.txt"))
    if not gt_cands:
        gt_cands = glob.glob(os.path.join(pair_path, "gt*.txt"))
    gt = gt_cands[0] if gt_cands else ""

    return {"next": nxt, "a11y": a11y, "request": req, "gt": gt}



# -----------------------------
# Main compute_score
# -----------------------------
def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    pred = (solution_str or "").strip()
    gt = (ground_truth or "").strip()
    if not pred or not gt:
        return {
            "score": 0.0,
            "reward/base": 0.0,
            "reward/len_penalty": 0.0,
            "reward/use_len_penalty": 0.0,
        }

    # 1) Base judge reward
    judge = GPTJudge(model=kwargs.get("judge_model", "gpt-4o-20241120"))
    verdict = judge.judge(pred=pred, gt=gt)
    scores = verdict.get("scores", {}) if isinstance(verdict, dict) else {}
    weights = kwargs.get("weights", DEFAULT_WEIGHTS)
    base_reward = _weighted_score(scores, weights)

    # 2) Length penalty
    use_lp = bool(kwargs.get("use_length_penalty", True))
    len_penalty = 0.0

    pred_tokens = gt_tokens = None
    length_ratio = None
    penalty_type = None

    if use_lp:
        len_penalty, pred_tokens, gt_tokens, length_ratio, penalty_type = _simple_asym_length_penalty_by_tokens(
            pred=pred,
            gt=gt,
            lower_ratio=float(kwargs.get("lower_ratio", 0.75)),
            upper_ratio=float(kwargs.get("upper_ratio", 1.25)),
            severe_max_penalty=float(kwargs.get("severe_max_penalty", 0.45)),
            mild_max_penalty=float(kwargs.get("mild_max_penalty", 0.25)),
            severe_alpha=float(kwargs.get("severe_alpha", 1.0)),
            no_short_penalty=bool(kwargs.get("no_short_penalty", False)),
            encoding_name=str(kwargs.get("encoding_name", "cl100k_base")),
        )

        # 可选：再额外算 token stats（保留你原来的逻辑）
        try:
            import tiktoken

            enc = tiktoken.get_encoding(str(kwargs.get("encoding_name", "cl100k_base")))
            pred_tokens = len(enc.encode(pred))
            gt_tokens = len(enc.encode(gt))
            length_ratio = (pred_tokens / max(gt_tokens, 1))
        except Exception:
            pass

    # 3) Action coherence reward (minimal intrusive)
    use_action_reward = bool(kwargs.get("use_action_reward", True))
    action_match = None
    action_pred = None
    files = None
    if use_action_reward:
        try:
            pair_path = extra_info.get("pair_path", ".") if isinstance(extra_info, dict) else "."
            files = _infer_pair_files(pair_path)

            # 关键文件缺失：直接 0
            if (not files.get("gt")) or (not os.path.exists(files["a11y"])) or (not os.path.exists(files["request"])) or (not os.path.exists(files["next"])):
                action_match = 0.0
            else:
                action_match = _action_coherence_reward(
                    textual_wm_response=pred,
                    img_path=files["next"],
                    gt_description_path=files["gt"],
                    a11y_path=files["a11y"],
                    instruction_path=files["request"],
                    **kwargs,
                )

                # 可选：为了 wandb 可读，额外把 pred action 拉出来（会多一次 LLM 调用）
                if bool(kwargs.get("log_action_pred", False)):
                    action_pred = _get_action(
                        img_path=None,
                        screen_description=pred,
                        screen_description_path=None,
                        a11y_path=files["a11y"],
                        instruction_path=files["request"],
                        **kwargs,
                    )
        except Exception:
            action_match = 0.0

    # 4) Mix rewards
    reward = base_reward - len_penalty

    if use_action_reward and action_match is not None:
        action_mix = float(kwargs.get("action_mix", 0.2))  # 0~1
        action_mix = 0.0 if action_mix < 0 else (1.0 if action_mix > 1 else action_mix)
        reward = (1.0 - action_mix) * reward + action_mix * float(action_match)

    # clamp
    reward = 0.0 if reward < 0.0 else (1.0 if reward > 1.0 else reward)

    # 5) Output (wandb-friendly)
    out = {
        "score": float(reward),
        "reward/base": float(base_reward),
        "reward/len_penalty": float(len_penalty),
        "reward/use_len_penalty": float(1.0 if use_lp else 0.0),
    }
    if pred_tokens is not None:
        out["reward/pred_tokens"] = float(pred_tokens)
    if gt_tokens is not None:
        out["reward/gt_tokens"] = float(gt_tokens)
    if length_ratio is not None:
        out["reward/length_ratio"] = float(length_ratio)
    if penalty_type is not None:
        out["reward/len_penalty_type"] = str(penalty_type)

    if use_action_reward:
        out["reward/use_action_reward"] = 1.0
        out["reward/files_pair_path"] = files.get("pair_path", "") if files else ""
        out["reward/files_next"] = files.get("next", "") if files else ""
        out["reward/files_a11y"] = files.get("a11y", "") if files else ""
        out["reward/files_request"] = files.get("request", "") if files else ""
        out["reward/files_gt"] = files.get("gt", "") if files else ""
        if action_match is not None:
            out["reward/action_match"] = float(action_match)

        # 最小侵入式：把 action_pred 的关键信息打到 wandb（兼容 list/dict 结构）
        if action_pred is not None:
            try:
                tc = None
                if isinstance(action_pred, list) and len(action_pred) == 1:
                    tc = action_pred[0].get("tool_call", {})
                elif isinstance(action_pred, dict):
                    tc = action_pred.get("tool_call", action_pred)
                tc = tc or {}
                out["reward/action_pred_function"] = str(tc.get("function", ""))
                out["reward/action_pred_status"] = str(tc.get("status", ""))
            except Exception:
                pass
    else:
        out["reward/use_action_reward"] = 0.0

    return out

if __name__ == "__main__":
    # Ensure DASHSCOPE_API_KEY is set in env.
    pred = 'This is Microsoft PowerPoint. The user has clicked on the \'Insert\' tab in the Ribbon, which is now active. The Main Editing Area has been updated to display a new slide with the text "Quarterly Report" centered. The sidebar has been updated to include a new panel labeled \'Design Ideas\'. The status bar has been updated to show the current slide number and the slide layout. The navigation area has been updated to display the new slide in the thumbnail view. The ribbon remains unchanged in terms of visible groups and controls.'
    gt = 'This is Microsoft PowerPoint. The user interaction was a single click on a control within the inserted Microsoft Forms object, which transitioned the embedded content from a selection screen to a different internal state. In the Next UI Screenshot, the Title Bar remains unchanged at the top with the same presentation name and window controls, and the Ribbon remains unchanged with the same tabs such as "Lêer", "Tuis", "Voeg in", "Ontwerp", and others visible and no tab switch indicated. The key change is in the Main Editing Area: the previously visible embedded Forms selection interface, which showed a choice-like layout with descriptive text and multiple option cards, has been replaced by a simplified Microsoft Forms placeholder or loading-like view. This new embedded view is centered on the slide, featuring a white background with a prominent teal horizontal header bar across the top edge of the embedded frame, the Microsoft Forms icon centered in the middle, and a small "Microsoft" label near the bottom center of the embedded area. The embedded object remains selected, indicated by white resize handles around its bounding box, but its internal content is now minimal and logo-focused rather than instructional. On the left, the Slide Navigation pane still shows a single slide thumbnail, but the thumbnail preview has updated to reflect the new embedded Forms appearance. On the right, the Sidebar remains the "Forms" pane with the header "Forms" and the section "My forms", showing buttons labeled "+ Nuwe vorm" and "+ Nuwe vraelys" and the list of forms such as "Titellose vorm", "Untitled quiz", and multiple "Untitled form" entries; this pane appears unchanged in content and position. The Status Bar at the bottom remains unchanged, still showing notes access, language, accessibility status, and the zoom level at "41%".'
    r = compute_score(
        data_source="ui_world_model_rl",
        solution_str=pred,
        ground_truth=gt,
        extra_info={"groundtruth_action": {"function": "click"}, 
                    "pair_path": "/path/to/cuwm/data_sample_3000/new_prompt/train_for_grpo/excel/bing_search/paired/excel_4_1454/pair_01"},
        use_length_penalty=True,
    )
    print("reward:", r)