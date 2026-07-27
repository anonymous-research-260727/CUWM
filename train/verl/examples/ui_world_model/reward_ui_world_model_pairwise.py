import os
import re
import math
import time
import json
import hashlib
import threading
from collections import Counter, OrderedDict
from typing import Any, Dict, Optional, List, Tuple

from examples.ui_world_model.cloudgpt_aoai import get_openai_client
from examples.ui_world_model.gpt_api import GPTJudge
from examples.ui_world_model.embedding import EmbeddingClient


# =========================
# 0) Config
# =========================

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

# final_reward = (w_judge*judge + w_sim*sim + w_len*len_score)/sum_w
DEFAULT_COMPONENT_WEIGHTS = {
    "judge": float(os.getenv("REWARD_W_JUDGE", "0.85")),
    "sim": float(os.getenv("REWARD_W_SIM", "0.1")),
    "len": float(os.getenv("REWARD_W_LEN", "0.05")),
}

# sim_score = (w_emb*emb_cos + w_rouge*rouge)/sum_w
DEFAULT_SIM_WEIGHTS = {
    "emb": float(os.getenv("REWARD_W_SIM_EMB", "0.0")),
    "rouge": float(os.getenv("REWARD_W_SIM_ROUGE", "1.0")),
}

DEFAULT_EMB_MODEL = os.getenv("REWARD_EMB_MODEL", "text-embedding-3-small")
DEFAULT_EMB_CACHE_SIZE = int(os.getenv("REWARD_EMB_CACHE_SIZE", "20000"))


# =========================
# 1) Small utils
# =========================

def _stable_hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        if p is None:
            p = ""
        if not isinstance(p, str):
            p = str(p)
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()

def _maybe_json_load(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", errors="ignore")
        except Exception:
            return None
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

def _weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    num = 0.0
    den = 0.0
    for k, w in weights.items():
        den += float(w)
        num += float(w) * float(scores.get(k, 0.0))
    return 0.0 if den <= 0 else (num / den)

def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


# =========================
# 2) Token length + SOFT length penalty
# =========================

def _count_tokens_tiktoken(text: str, encoding_name: str = "cl100k_base") -> int:
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)

def _soft_ratio_length_penalty_by_tokens(
    pred: str,
    gt: str,
    lower_ratio: float = 0.75,
    upper_ratio: float = 1.25,
    max_penalty: float = 0.25,
    alpha: float = 2.0,                 # 越大惩罚上升越快
    encoding_name: str = "cl100k_base",
) -> Tuple[float, int, int, float]:
    """
    gap:
      pred < lo: (lo-pred)/lo
      pred > hi: (pred-hi)/hi
      else: 0
    penalty (soft):
      max_penalty * (1 - exp(-alpha*gap))
    """
    gt_tokens = _count_tokens_tiktoken(gt, encoding_name=encoding_name)
    pred_tokens = _count_tokens_tiktoken(pred, encoding_name=encoding_name)
    ratio = pred_tokens / max(gt_tokens, 1)

    if gt_tokens <= 0 or max_penalty <= 0:
        return 0.0, pred_tokens, gt_tokens, ratio

    lo = max(1, int(lower_ratio * gt_tokens))
    hi = max(lo + 1, int(upper_ratio * gt_tokens))

    if lo <= pred_tokens <= hi:
        return 0.0, pred_tokens, gt_tokens, ratio

    if pred_tokens < lo:
        gap = (lo - pred_tokens) / float(lo)
    else:
        gap = (pred_tokens - hi) / float(hi)

    pen = max_penalty * (1.0 - math.exp(-alpha * gap))
    return float(pen), pred_tokens, gt_tokens, ratio


# =========================
# 3) Similarity: ROUGE (relative) + Embedding cosine (API only)
# =========================

_WORD_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")

def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _WORD_RE.findall(text.strip())

def _f1(p: float, r: float) -> float:
    if p <= 0.0 or r <= 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)

def _rouge1_f1(pred_toks: List[str], gt_toks: List[str]) -> float:
    if not pred_toks or not gt_toks:
        return 0.0
    pc = Counter(pred_toks)
    gc = Counter(gt_toks)
    overlap = sum(min(v, gc.get(k, 0)) for k, v in pc.items())
    prec = overlap / max(1, len(pred_toks))
    rec = overlap / max(1, len(gt_toks))
    return float(_f1(prec, rec))

def _lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]

def _rougeL_f1(pred_toks: List[str], gt_toks: List[str]) -> float:
    if not pred_toks or not gt_toks:
        return 0.0
    lcs = _lcs_len(pred_toks, gt_toks)
    prec = lcs / max(1, len(pred_toks))
    rec = lcs / max(1, len(gt_toks))
    return float(_f1(prec, rec))

def _rouge_relative_similarity(pred: str, gt: str, w_r1: float = 0.5, w_rl: float = 0.5) -> float:
    pt = _tokenize(pred)
    gtoks = _tokenize(gt)
    r1 = _rouge1_f1(pt, gtoks)
    rl = _rougeL_f1(pt, gtoks)
    s = (w_r1 * r1 + w_rl * rl) / max(1e-8, (w_r1 + w_rl))
    return _clamp01(s)

def _cosine(u: List[float], v: List[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu <= 0.0 or nv <= 0.0:
        return 0.0
    return float(dot / (math.sqrt(nu) * math.sqrt(nv)))



from typing import Dict

def _similarity_score_api_only(
    pred: str,
    gt: str,
    emb_client: EmbeddingClient,
    sim_weights: Dict[str, float],
    rouge_w_r1: float = 0.5,
    rouge_w_rl: float = 0.5,
) -> Dict[str, float]:
    w_emb = float(sim_weights.get("emb", 0.5))
    w_rouge = float(sim_weights.get("rouge", 0.5))

    # 先给默认值（没算就保持 0）
    rouge_sim = 0.0
    emb_cos = 0.0

    # 仅在权重 > 0 时计算 ROUGE
    if w_rouge > 0.0:
        rouge_sim = float(_rouge_relative_similarity(pred, gt, w_r1=rouge_w_r1, w_rl=rouge_w_rl))

    # 仅在权重 > 0 时计算 embedding cosine（避免 API）
    if w_emb > 0.0:
        v1 = emb_client.embed(pred)
        v2 = emb_client.embed(gt)
        emb_cos_raw = _cosine(v1, v2)
        emb_cos = float(_clamp01((emb_cos_raw + 1.0) / 2.0))  # 映射到[0,1]

    denom = w_emb + w_rouge
    if denom <= 0.0:
        # 两个权重都为 0：定义一个稳定默认（你也可以选择 raise）
        sim = 0.0
    else:
        sim = (w_emb * emb_cos + w_rouge * rouge_sim) / denom
        sim = float(_clamp01(sim))

    return {
        "reward/sim_score": float(sim),
        "reward/sim_emb": float(emb_cos),
        "reward/sim_rouge": float(rouge_sim),
    }



# =========================
# 4) Singletons (minimal)
# =========================

_JUDGE = None
_EMB = None
_SINGLETON_LOCK = threading.Lock()

def _get_judge(model: str):
    global _JUDGE
    with _SINGLETON_LOCK:
        if _JUDGE is None or getattr(_JUDGE, "model", None) != model:
            _JUDGE = GPTJudge(model=model)
    return _JUDGE

def _get_emb(model: str):
    global _EMB
    with _SINGLETON_LOCK:
        if _EMB is None or getattr(_EMB, "model", None) != model:
            _EMB = EmbeddingClient(model=model)
    return _EMB


# =========================
# 5) compute_score
# =========================

def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    pred = (solution_str or "").strip()
    gt = (ground_truth or "").strip()
    if not pred or not gt:
        return {
            "score": 0.0,
            "reward/judge_score": 0.0,
            "reward/sim_score": 0.0,
            "reward/len_score": 0.0,
            "reward/len_penalty_soft": 0.0,
            "reward/use_len_penalty": 0.0,
        }

    _ = _maybe_json_load(extra_info) or {}

    # ---- 1) LLM Judge ----
    judge_model = str(kwargs.get("judge_model", "gpt-4o-20241120"))
    judge = _get_judge(judge_model)

    verdict = judge.judge(pred=pred, gt=gt)
    scores = verdict.get("scores", {}) if isinstance(verdict, dict) else {}
    judge_weights = kwargs.get("weights", DEFAULT_WEIGHTS)

    judge_score = _clamp01(_weighted_score(scores, judge_weights))

    # ---- 2) Similarity (API-only embedding + rouge) ----
    emb_model = str(kwargs.get("emb_model", DEFAULT_EMB_MODEL))
    emb_client = _get_emb(emb_model)

    sim_weights = kwargs.get("sim_weights", DEFAULT_SIM_WEIGHTS)
    sim_metrics = _similarity_score_api_only(
        pred=pred,
        gt=gt,
        emb_client=emb_client,
        sim_weights=sim_weights,
        rouge_w_r1=float(kwargs.get("rouge_w_r1", 0.5)),
        rouge_w_rl=float(kwargs.get("rouge_w_rl", 0.5)),
    )
    sim_score = float(sim_metrics["reward/sim_score"])

    # ---- 3) Soft length penalty -> len_score ----
    use_lp = bool(kwargs.get("use_length_penalty", True))
    max_penalty = float(kwargs.get("max_penalty", 0.25))
    len_penalty = 0.0
    pred_tokens = None
    gt_tokens = None
    length_ratio = None
    len_score = 1.0

    if use_lp and max_penalty > 0:
        len_penalty, pred_tokens, gt_tokens, length_ratio = _soft_ratio_length_penalty_by_tokens(
            pred=pred,
            gt=gt,
            lower_ratio=float(kwargs.get("lower_ratio", 0.75)),
            upper_ratio=float(kwargs.get("upper_ratio", 1.25)),
            max_penalty=max_penalty,
            alpha=float(kwargs.get("len_penalty_alpha", 2.0)),
            encoding_name=str(kwargs.get("encoding_name", "cl100k_base")),
        )
        len_score = _clamp01(1.0 - (len_penalty / max_penalty))
    else:
        use_lp = False
        len_score = 1.0

    # ---- 4) Combine with weights ----
    comp_w = kwargs.get("component_weights", DEFAULT_COMPONENT_WEIGHTS)
    wj = float(comp_w.get("judge"))
    ws = float(comp_w.get("sim"))
    wl = float(comp_w.get("len"))
    denom = max(1e-8, (wj + ws + wl))

    reward = (wj * judge_score + ws * sim_score + wl * len_score) / denom
    reward = _clamp01(reward)

    # ---- 5) Return dict (verl NaiveRewardManager consumes "score") ----
    out = {
        "score": float(reward),

        "reward/judge_score": float(judge_score),
        "reward/sim_score": float(sim_score),
        "reward/len_score": float(len_score),

        "reward/len_penalty_soft": float(len_penalty),
        "reward/use_len_penalty": float(1.0 if use_lp else 0.0),
        "reward/max_penalty": float(max_penalty),

        "reward/w_judge": float(wj),
        "reward/w_sim": float(ws),
        "reward/w_len": float(wl),
        "reward/w_sim_emb": float(sim_weights.get("emb", 0.5)),
        "reward/w_sim_rouge": float(sim_weights.get("rouge", 0.5)),
    }

    # token stats (optional)
    if pred_tokens is not None:
        out["reward/pred_tokens"] = float(pred_tokens)
    if gt_tokens is not None:
        out["reward/gt_tokens"] = float(gt_tokens)
    if length_ratio is not None:
        out["reward/length_ratio"] = float(length_ratio)

    # similarity breakdown
    out.update(sim_metrics)

    # optional: expand judge per-aspect scores
    for k, v in scores.items():
        try:
            out[f"judge/{k}"] = float(v)
        except Exception:
            pass

    return out

# import os
# import json
# import time
# import hashlib
# import re
# from typing import Any, Dict, Optional, Tuple

# from concurrent.futures import ThreadPoolExecutor, as_completed
# from openai import OpenAI
# from examples.ui_world_model.gpt_api import GPTJudge


# DEFAULT_WEIGHTS = {
#     "app_name": 0.8,
#     "user_action": 1.4,
#     "title_bar": 1.0,
#     "ribbon": 1.1,
#     "main_editing_area": 1.5,
#     "sidebar_pane": 0.8,
#     "navigation_area": 0.6,
#     "status_bar": 0.8,
# }


# def _weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
#     num = 0.0
#     den = 0.0
#     for k, w in weights.items():
#         den += float(w)
#         num += float(w) * float(scores.get(k, 0.0))
#     if den <= 0:
#         return 0.0
#     return num / den

# def _count_tokens_tiktoken(text: str, encoding_name: str = "cl100k_base") -> int:
#     """
#     Count tokens using tiktoken. Works well as a proxy length measure.
#     """
#     if not text:
#         return 0
#     try:
#         import tiktoken
#         enc = tiktoken.get_encoding(encoding_name)
#         return len(enc.encode(text))
#     except Exception:
#         # Fallback: rough estimate if tiktoken unavailable
#         # assume ~4 chars per token for English-ish text; more conservative for mixed text
#         return max(1, len(text) // 4)

# def _ratio_length_penalty_by_tokens(
#     pred: str,
#     gt: str,
#     lower_ratio: float = 0.75,
#     upper_ratio: float = 1.25,
#     max_penalty: float = 0.25,
#     encoding_name: str = "cl100k_base",
# ) -> float:
#     """
#     If pred tokens are outside [lower_ratio * gt_tokens, upper_ratio * gt_tokens],
#     apply a penalty that increases with how far it is outside the band.
#     Penalty is capped at max_penalty.
#     """
#     gt_tokens = _count_tokens_tiktoken(gt, encoding_name=encoding_name)
#     pred_tokens = _count_tokens_tiktoken(pred, encoding_name=encoding_name)

#     # If GT is extremely short, avoid unstable ratios; treat as no penalty banding.
#     if gt_tokens <= 0:
#         return 0.0

#     lo = max(1, int(lower_ratio * gt_tokens))
#     hi = max(lo + 1, int(upper_ratio * gt_tokens))

#     if lo <= pred_tokens <= hi:
#         return 0.0

#     # How far outside the band, normalized by band edge (scale-invariant)
#     if pred_tokens < lo:
#         # e.g. pred=50, lo=80 -> gap=30/80=0.375
#         gap = (lo - pred_tokens) / float(lo)
#     else:
#         gap = (pred_tokens - hi) / float(hi)

#     # Map gap -> penalty (smooth-ish). Linear is fine for shaping.
#     pen = max_penalty * min(1.0, gap)  # cap when gap>=1 (2x outside)
#     return float(pen)


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


# def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
#     """
#     GRPO reward for UI world-model textual prediction:
#     - Uses Qwen LLM Judge to score PRED vs GT (both text).
#     - Returns a scalar in [0, 1] (with small penalties applied).

#     Expected:
#       - solution_str: model output (PRED)
#       - ground_truth: GT text description of next frame (teacher label)
#       - extra_info: may contain action/prev/next etc., but judge is text-only here.

#     kwargs (optional):
#       - judge_model: str, default "qwen-plus"
#       - enable_thinking: bool, default False (recommend False for speed)
#       - weights: dict, override DEFAULT_WEIGHTS
#       - length_penalty: bool, default True
#     """
#     pred = (solution_str or "").strip()
#     gt = (ground_truth or "").strip()
#     if not pred or not gt:
#         return 0.0

#     # Parse extra_info if you later want to add heuristics; keep for compatibility
#     _ = _maybe_json_load(extra_info) or {}

#     judge_model = kwargs.get("judge_model", "qwen-plus")
#     enable_thinking = bool(kwargs.get("enable_thinking", False))
#     thinking_budget = int(kwargs.get("thinking_budget", 0))
#     api_key = kwargs.get("api_key", "<YOUR_API_KEY>")
#     # base_url = kwargs.get("base_url", "https://api.vectorengine.ai")
#     base_url = kwargs.get("base_url", "https://api.vectorengine.ai/v1")
#     # base_url = kwargs.get("base_url", "https://api.vectorengine.ai/v1/chat/completions")

#     judge = _get_judge(
#         model=judge_model,
#         enable_thinking=enable_thinking,
#         thinking_budget=thinking_budget,
#         api_key=api_key,
#         base_url=base_url,
#         temperature=float(kwargs.get("temperature", 0.0)),
#         top_p=float(kwargs.get("top_p", 1.0)),
#         max_retries=int(kwargs.get("max_retries", 2)),
#         backoff_sec=float(kwargs.get("backoff_sec", 0.8)),
#         max_cache_size=int(kwargs.get("max_cache_size", 50000)),
#     )

#     verdict = judge.judge(pred=pred, gt=gt)
#     scores = verdict.get("scores", {}) if isinstance(verdict, dict) else {}
#     weights = kwargs.get("weights", DEFAULT_WEIGHTS)

#     reward = _weighted_score(scores, weights)

#     # Small shaping to reduce rambling (which increases hallucination risk)
#     if bool(kwargs.get("use_length_penalty", True)):
#         reward -= _ratio_length_penalty_by_tokens(
#             pred=pred,
#             gt=gt,
#             lower_ratio=float(kwargs.get("lower_ratio", 0.75)),
#             upper_ratio=float(kwargs.get("upper_ratio", 1.25)),
#             max_penalty=float(kwargs.get("max_penalty", 0.25)),
#             encoding_name=str(kwargs.get("encoding_name", "cl100k_base")),
#         )

#     # Clamp
#     if reward < 0.0:
#         reward = 0.0
#     if reward > 1.0:
#         reward = 1.0
#     return float(reward)


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
