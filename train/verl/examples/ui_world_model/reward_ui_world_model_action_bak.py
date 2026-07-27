import os
import json
import time
import hashlib
import re
from typing import Any, Dict, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from examples.ui_world_model.gpt_api import GPTJudge
from examples.ui_world_model.action_eval_a11y import ActionEvaluationA11y

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
        # assume ~4 chars per token for English-ish text; more conservative for mixed text
        return max(1, len(text) // 4)

def _get_dashscope_client() -> OpenAI:
    # 你也可以改成 thread-local；这里先最小可用
    return OpenAI(
        api_key="<YOUR_API_KEY>",
        base_url="https://api.vectorengine.ai/v1",
    )

def _generate_option_generation_prompt(
    num_options: int,
    app: str,
    folder: str,
):
    save_prompt_path = os.path.join(folder, "option_generation_prompt.txt")      
    from prompts import (
        ACTION_PREDICTION_A11Y_SYS_PROMPT_GPT,
        ACTION_PREDICTION_A11Y_USER_PROMPT_GPT,
        SUPPORTED_ACTIONS,
    )
    
    supported_actions = SUPPORTED_ACTIONS[app]
    sys_prompt = ACTION_PREDICTION_A11Y_SYS_PROMPT_GPT.replace("{num_options}", str(num_options))
    
    with open(os.path.join(folder, "request.txt"), "r", encoding="utf-8") as f:
        instruction = f.read().strip()
    
    with open(os.path.join(folder, "a11y.json"), "r", encoding="utf-8") as f:
        a11y = json.load(f)
        
    # print(a11y)
    
    usr_prompt = ACTION_PREDICTION_A11Y_USER_PROMPT_GPT.format(
        instruction=instruction,
        a11y=json.dumps(a11y, indent=2),
        actions=supported_actions,
        num_options=num_options,
    )
    
    prompt = sys_prompt + "\n\n" + usr_prompt
        
    with open(save_prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
        # self.logger.info(f"Option generation prompt saved to {save_prompt_path}")
        
    return prompt

    
def _get_app(folder):
    if "word" in folder.lower():
        app = "word"
    elif "excel" in folder.lower():
        app = "excel"
    elif "ppt" in folder.lower():
        app = "ppt"
    else:
        raise ValueError(f"Cannot infer app type from folder: {folder}")
    return app

def _qwen_generate_action(folder, **kwargs):
    """
    调 DashScope（OpenAI compatible）生成 action JSON
    """
    client = _get_dashscope_client()
    model = kwargs.get("qwen_action_model", "qwen3-vl-flash")  # 你也可换成纯文本模型
    temperature = float(kwargs.get("qwen_action_temperature", 0.0))
    top_p = float(kwargs.get("qwen_action_top_p", 1.0))

    save_path = os.path.join(folder, "action_options.json")
    raw_save_path = os.path.join(folder, "action_options_raw.json")
    if os.path.exists(save_path):
        # self.logger.info(f"Action options already exist at {save_path}, loading...")
        with open(save_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    app = _get_app(folder)
    option_generation_prompt = _generate_option_generation_prompt(
        num_options=1,
        app=app,
        folder=folder,
    )
    import pdb;pdb.set_trace()
    image_path = os.path.join(folder, "prev.png")
    annotated_image_path = os.path.join(folder, "prev_annotated.png")

    completion = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": option_generation_prompt}],
            "images": [
                {"type": "image_url", "image_url": {"url": f"{image_path}"}},
                {"type": "image_url", "image_url": {"url": f"{annotated_image_path}"}},
            ],
        }],
        temperature=temperature,
        top_p=top_p,
        stream=False
    )
    out_txt = completion.choices[0].message.content
    response = out_txt.strip()
    # jsonify and save
    import re
    # Find the first JSON list
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
        response_json_str = match.group(0)
    else:
        response_json_str = response.replace("```json", "").replace("```", "").strip()
    
    try:
        response_json = json.loads(response_json_str)
    except json.JSONDecodeError:
        try:
            import ast
            response_json = ast.literal_eval(response_json_str)
        except Exception:
            raise
        
    with open(raw_save_path, "w", encoding="utf-8") as f:
        json.dump(response_json, f, indent=2)
        # self.logger.info(f"Action options raw saved to {save_path}")
    
    # process actions
    action_options = []
    for ele in response_json:
        action_option = ele["tool_call"]
        if "control_label" in action_option["args"]:
            control_label = action_option["args"]["control_label"]
            with open(os.path.join(folder, "a11y.json"), "r", encoding="utf-8") as f:
                a11y = json.load(f)
                if type(control_label) == int and 1 <= control_label <= len(a11y):
                    control_info = a11y[control_label - 1]
                    assert control_info["label"] == control_label
                    del control_info["control_rect"]
                    del control_info["source"]
                    del control_info["label"]
                    action_option["args"]["control_info"] = control_info
            del action_option["args"]["control_label"]
        action_options.append(action_option)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(action_options, f, indent=2)
    return response_json


    return score

def _simple_asym_length_penalty_by_tokens(
    pred: str,
    gt: str,
    lower_ratio: float = 0.75,
    upper_ratio: float = 1.25,
    # 严厉惩罚的上限（超长）
    severe_max_penalty: float = 0.45,
    # 轻微惩罚的上限（过短）
    mild_max_penalty: float = 0.25,
    # 严厉惩罚“长得多快”（越大越狠）
    severe_alpha: float = 1.0,
    # 过短是否不惩罚（你说“不用严厉惩罚”，这里给你一个开关）
    no_short_penalty: bool = False,
    encoding_name: str = "cl100k_base",
) -> Tuple[float, Optional[int], Optional[int], Optional[float], str]:
    """
    简单非对称长度惩罚（token 比例）：
    - pred/gt > upper_ratio: 严厉惩罚（默认二次增长，cap 到 severe_max_penalty）
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
        # 线性：ratio 从 lower_ratio 降到 0，penalty 从 0 增到 mild_max_penalty
        gap = (lower_ratio - ratio) / max(lower_ratio, 1e-6)  # 0~(理论上>1)
        pen = mild_max_penalty * min(1.0, max(0.0, gap))
        return float(pen), pred_tokens, gt_tokens, ratio, "short_mild"

    # 超长：严厉惩罚
    # ratio 从 upper_ratio 往上走，gap=0开始；用二次（或 severe_alpha 次）增长更狠
    gap = (ratio - upper_ratio) / max(upper_ratio, 1e-6)  # 0~...
    shaped = min(1.0, max(0.0, gap)) ** float(severe_alpha)
    pen = severe_max_penalty * shaped
    return float(pen), pred_tokens, gt_tokens, ratio, "long_severe"

def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    pred = (solution_str or "").strip()
    gt = (ground_truth or "").strip()
    if not pred or not gt:
        return {"score": 0.0, "reward/base": 0.0, "reward/len_penalty": 0.0, "reward/use_len_penalty": 0.0}

    judge = GPTJudge(model="gpt-4o-20241120")
    
    verdict = judge.judge(pred=pred, gt=gt)
    scores = verdict.get("scores", {}) if isinstance(verdict, dict) else {}
    weights = kwargs.get("weights", DEFAULT_WEIGHTS)

    base_reward = _weighted_score(scores, weights)

    use_lp = bool(kwargs.get("use_length_penalty", True))
    len_penalty = 0.0

    # 如果你想把 token 数也打出来：复制一份 token 计数逻辑（别在 penalty 函数里“黑盒”掉）
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

        # 2) 再额外算 token stats（可选，但很建议）
        try:
            import tiktoken
            enc = tiktoken.get_encoding(str(kwargs.get("encoding_name", "cl100k_base")))
            pred_tokens = len(enc.encode(pred))
            gt_tokens = len(enc.encode(gt))
            length_ratio = (pred_tokens / max(gt_tokens, 1))
        except Exception:
            pass


    # -----------------------------
    # NEW: action-match reward (qwen api)
    # -----------------------------
    use_action_reward = True
    action_match = None
    action_pred = None

    if use_action_reward:
        import pdb;pdb.set_trace()
        try:
            action_pred = _qwen_generate_action(folder=extra_info.get("pair_path", "."), **kwargs)
            if isinstance(action_pred, dict):
                action_match = 1.0 if _actions_equal(action_pred, gt_action, **kwargs) else 0.0
            else:
                action_match = 0.0
        except Exception:
            action_match = 0.0


    reward = base_reward - len_penalty

    # 把 action_match 融合进 reward（最小侵入式，且保持 [0,1]）
    if use_action_reward and action_match is not None:
        action_mix = float(kwargs.get("action_mix", 0.2))  # 0~1，越大越看重 action
        action_mix = 0.0 if action_mix < 0 else (1.0 if action_mix > 1 else action_mix)
        reward = (1.0 - action_mix) * reward + action_mix * float(action_match)

    # clamp
    reward = 0.0 if reward < 0.0 else (1.0 if reward > 1.0 else reward)


    # ✅ 关键：返回 dict，并且必须包含 "score" 这个 key（NaiveRewardManager 用它当最终 reward）
    # 下面这些都会进入 reward_extra_info -> reduce -> wandb
    out = {
        "score": float(reward),
        "reward/base": float(base_reward),
        "reward/len_penalty": float(len_penalty),
        "reward/use_len_penalty": float(1.0 if use_lp else 0.0),
    }
    if pred_tokens is not None: out["reward/pred_tokens"] = float(pred_tokens)
    if gt_tokens is not None: out["reward/gt_tokens"] = float(gt_tokens)
    if length_ratio is not None: out["reward/length_ratio"] = float(length_ratio)
    if penalty_type is not None: out["reward/len_penalty_type"] = str(penalty_type)

    if use_action_reward:
        out["reward/use_action_reward"] = 1.0
        if action_match is not None:
            out["reward/action_match"] = float(action_match)
        if action_pred is not None:
            # 不建议把整个 action_pred 打到 wandb（可能很大/含敏感）
            # 这里只打几个关键信息
            out["reward/action_pred_function"] = str(action_pred.get("function", ""))
            out["reward/action_pred_control_name"] = str(action_pred.get("control_name", ""))
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
