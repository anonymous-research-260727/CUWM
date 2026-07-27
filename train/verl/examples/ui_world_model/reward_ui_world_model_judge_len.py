import os
import json
import time
import hashlib
import re
from typing import Any, Dict, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
# from examples.ui_world_model.gpt_api import GPTJudge
from examples.ui_world_model.qwen_api import QwenJudge  


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

def _ratio_length_penalty_by_tokens(
    pred: str,
    gt: str,
    lower_ratio: float = 0.75,
    upper_ratio: float = 1.25,
    max_penalty: float = 0.25,
    encoding_name: str = "cl100k_base",
) -> float:
    """
    If pred tokens are outside [lower_ratio * gt_tokens, upper_ratio * gt_tokens],
    apply a penalty that increases with how far it is outside the band.
    Penalty is capped at max_penalty.
    """
    gt_tokens = _count_tokens_tiktoken(gt, encoding_name=encoding_name)
    pred_tokens = _count_tokens_tiktoken(pred, encoding_name=encoding_name)

    # If GT is extremely short, avoid unstable ratios; treat as no penalty banding.
    if gt_tokens <= 0:
        return 0.0

    lo = max(1, int(lower_ratio * gt_tokens))
    hi = max(lo + 1, int(upper_ratio * gt_tokens))

    if lo <= pred_tokens <= hi:
        return 0.0

    # How far outside the band, normalized by band edge (scale-invariant)
    if pred_tokens < lo:
        # e.g. pred=50, lo=80 -> gap=30/80=0.375
        gap = (lo - pred_tokens) / float(lo)
    else:
        gap = (pred_tokens - hi) / float(hi)

    # Map gap -> penalty (smooth-ish). Linear is fine for shaping.
    pen = max_penalty * min(1.0, gap)  # cap when gap>=1 (2x outside)
    return float(pen)


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    """
    GRPO reward for UI world-model textual prediction:
    - Uses Qwen LLM Judge to score PRED vs GT (both text).
    - Returns a scalar in [0, 1] (with small penalties applied).

    Expected:
      - solution_str: model output (PRED)
      - ground_truth: GT text description of next frame (teacher label)
      - extra_info: may contain action/prev/next etc., but judge is text-only here.

    kwargs (optional):
      - judge_model: str, default "qwen-plus"
      - enable_thinking: bool, default False (recommend False for speed)
      - weights: dict, override DEFAULT_WEIGHTS
      - length_penalty: bool, default True
    """
    pred = (solution_str or "").strip()
    gt = (ground_truth or "").strip()
    if not pred or not gt:
        return 0.0

    # Parse extra_info if you later want to add heuristics; keep for compatibility

    judge_model = kwargs.get("judge_model", "qwen-plus")
    enable_thinking = bool(kwargs.get("enable_thinking", False))
    thinking_budget = int(kwargs.get("thinking_budget", 0))
    api_key = kwargs.get("api_key", "<YOUR_API_KEY>")
    # base_url = kwargs.get("base_url", "https://api.vectorengine.ai")
    base_url = kwargs.get("base_url", "https://api.vectorengine.ai/v1")
    # base_url = kwargs.get("base_url", "https://api.vectorengine.ai/v1/chat/completions")

    judge = QwenJudge(
        model=judge_model,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        api_key=api_key,
        base_url=base_url,
        temperature=float(kwargs.get("temperature", 0.0)),
        top_p=float(kwargs.get("top_p", 1.0)),
        max_retries=int(kwargs.get("max_retries", 2)),
        backoff_sec=float(kwargs.get("backoff_sec", 0.8)),
        max_cache_size=int(kwargs.get("max_cache_size", 50000)),
    )

    verdict = judge.judge(pred=pred, gt=gt)
    scores = verdict.get("scores", {}) if isinstance(verdict, dict) else {}
    weights = kwargs.get("weights", DEFAULT_WEIGHTS)

    base_reward = _weighted_score(scores, weights)

    use_lp = bool(kwargs.get("use_length_penalty", True))
    len_penalty = 0.0

    # 如果你想把 token 数也打出来：复制一份 token 计数逻辑（别在 penalty 函数里“黑盒”掉）
    pred_tokens = gt_tokens = None
    length_ratio = None

    if use_lp:
        # 1) 先算 penalty
        len_penalty = _ratio_length_penalty_by_tokens(
            pred=pred,
            gt=gt,
            lower_ratio=float(kwargs.get("lower_ratio", 0.75)),
            upper_ratio=float(kwargs.get("upper_ratio", 1.25)),
            max_penalty=float(kwargs.get("max_penalty", 0.25)),
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

    reward = base_reward - len_penalty

    # clamp
    reward = 0.0 if reward < 0.0 else (1.0 if reward > 1.0 else reward)

    # ✅ 关键：返回 dict，并且必须包含 "score" 这个 key（NaiveRewardManager 用它当最终 reward）
    out = {
        "score": float(reward),

        # 下面这些都会进入 reward_extra_info -> reduce -> wandb
        "reward/base": float(base_reward),
        "reward/len_penalty": float(len_penalty),
        "reward/use_len_penalty": float(1.0 if use_lp else 0.0),
    }
    if pred_tokens is not None: out["reward/pred_tokens"] = float(pred_tokens)
    if gt_tokens is not None: out["reward/gt_tokens"] = float(gt_tokens)
    if length_ratio is not None: out["reward/length_ratio"] = float(length_ratio)

    return out


# -----------------------------
# 6) Minimal local test
# -----------------------------
if __name__ == "__main__":
    # Ensure DASHSCOPE_API_KEY is set in env.
    pred = 'This is Microsoft PowerPoint. The user has clicked on the \'Insert\' tab in the Ribbon, which is now active. The Main Editing Area has been updated to display a new slide with the text "Quarterly Report" centered. The sidebar has been updated to include a new panel labeled \'Design Ideas\'. The status bar has been updated to show the current slide number and the slide layout. The navigation area has been updated to display the new slide in the thumbnail view. The ribbon remains unchanged in terms of visible groups and controls.'
    gt = 'This is Microsoft PowerPoint. The user interaction was a single click on a control within the inserted Microsoft Forms object, which transitioned the embedded content from a selection screen to a different internal state. In the Next UI Screenshot, the Title Bar remains unchanged at the top with the same presentation name and window controls, and the Ribbon remains unchanged with the same tabs such as "Lêer", "Tuis", "Voeg in", "Ontwerp", and others visible and no tab switch indicated. The key change is in the Main Editing Area: the previously visible embedded Forms selection interface, which showed a choice-like layout with descriptive text and multiple option cards, has been replaced by a simplified Microsoft Forms placeholder or loading-like view. This new embedded view is centered on the slide, featuring a white background with a prominent teal horizontal header bar across the top edge of the embedded frame, the Microsoft Forms icon centered in the middle, and a small "Microsoft" label near the bottom center of the embedded area. The embedded object remains selected, indicated by white resize handles around its bounding box, but its internal content is now minimal and logo-focused rather than instructional. On the left, the Slide Navigation pane still shows a single slide thumbnail, but the thumbnail preview has updated to reflect the new embedded Forms appearance. On the right, the Sidebar remains the "Forms" pane with the header "Forms" and the section "My forms", showing buttons labeled "+ Nuwe vorm" and "+ Nuwe vraelys" and the list of forms such as "Titellose vorm", "Untitled quiz", and multiple "Untitled form" entries; this pane appears unchanged in content and position. The Status Bar at the bottom remains unchanged, still showing notes access, language, accessibility status, and the zoom level at "41%".'
    r = compute_score(
        data_source="ui_world_model_rl",
        solution_str=pred,
        ground_truth=gt,
        extra_info=json.dumps({"action": {"function": "click"}}),
        judge_model="qwen-plus",           # or "qwen-turbo" for cheaper/faster
        enable_thinking=False,            # recommend False for reward
        use_length_penalty=True,
    )
    print("reward:", r)
