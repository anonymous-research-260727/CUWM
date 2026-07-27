import os
import json
import time
import hashlib
import re
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

# -----------------------------
# 1) Judge Prompt
# -----------------------------
TEXTUAL_WM_EVAL_PROMPT = """
You are an impartial LLM-as-a-Judge. Your task is to grade a model prediction (PRED) against the ground truth (GT) for describing the “Next UI Screenshot” of an Office application (e.g., Microsoft Word).

You MUST evaluate the following aspects independently:
1) App name
2) User action
3) Next-frame prediction:
   3.1) Title Bar
   3.2) Ribbon
   3.3) Main Editing Area / Canvas
   3.4) Sidebar / Pane
   3.5) Navigation Area
   3.6) Status Bar

Scoring rule for EACH aspect (use ONLY these values):
- 0   = completely incorrect / contradicts GT / missing when GT contains it
- 0.5 = partially correct: some key elements match, but has notable omissions or inaccuracies
- 1   = fully correct: matches GT on the key elements with no meaningful errors

Critical evaluation guidelines:
- Use GT as the single source of truth.
- Judge content fidelity, not writing quality.
- Be strict about factual UI elements (active tab name, document title, zoom %, panes open/closed, specific text edits).
- Penalize hallucinations: if PRED adds UI changes or elements not supported by GT, deduct in the relevant aspect(s).
- If GT does NOT mention a sub-area (e.g., Navigation Area), then:
  - If PRED also does not mention it → score 1 (no contradiction).
  - If PRED claims a specific change/state that GT does not support → score 0.5 or 0 depending on how strong/incorrect it is.
- When scoring 0.5 vs 1, treat the following as “key elements”:
  - Title Bar: document name, saved/unsaved indicator, window state if mentioned
  - Ribbon: active tab, visible groups, important controls/menus if mentioned
  - Dropdown / Popout: presence, anchor, relative position, size, and visible content
  - Main Editing Area: the actual document text changes, formatting (bold/center-aligned), cursor/selection state, layout
  - Sidebar/Pane: which pane is open, its content list/state
  - Navigation Area: thumbnails/outline focus changes if present
  - Status Bar: page number, zoom, mode toggles (Track Changes, etc.)

Output format requirements:
- Output ONLY valid JSON.
- No markdown, no extra text.
- Include per-aspect scores.

Return JSON with exactly this structure:
{{
  "scores": {{
    "app_name": <0|0.5|1>,
    "user_action": <0|0.5|1>,
    "title_bar": <0|0.5|1>,
    "ribbon": <0|0.5|1>,
    "main_editing_area": <0|0.5|1>,
    "sidebar_pane": <0|0.5|1>,
    "navigation_area": <0|0.5|1>,
    "status_bar": <0|0.5|1>
  }},
  "notes": {{
    "app_name": "<one short sentence rationale>",
    "user_action": "<one short sentence rationale>",
    "title_bar": "<one short sentence rationale>",
    "ribbon": "<one short sentence rationale>",
    "main_editing_area": "<one short sentence rationale>",
    "sidebar_pane": "<one short sentence rationale>",
    "navigation_area": "<one short sentence rationale>",
    "status_bar": "<one short sentence rationale>"
  }}
}}

Now perform the evaluation.

PRED:
<<<
{PRED}
>>>

GT:
<<<
{GT}
>>>
""".strip()


# -----------------------------
# 2) Utilities
# -----------------------------
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
        # try to extract a JSON object substring
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


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


def _extract_json_object(text: str) -> Optional[str]:
    """Best-effort: pull the first top-level JSON object out of a messy response."""
    if not text:
        return None
    text = text.strip()

    # If it's already valid JSON
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Grab the first {...} block
    start = text.find("{")
    if start < 0:
        return None

    # Scan braces to find a balanced object
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1].strip()
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    return candidate  # still return; caller may do repair
    return None


def _coerce_score(v: Any) -> float:
    # Allowed: 0, 0.5, 1. Coerce safely.
    try:
        fv = float(v)
    except Exception:
        return 0.0
    # snap to nearest allowed
    if fv <= 0.25:
        return 0.0
    if fv <= 0.75:
        return 0.5
    return 1.0


# -----------------------------
# 3) Qwen Judge Client (OpenAI compatible)
# -----------------------------
class QwenJudge:
    """
    Thin wrapper with retry + cache.
    Designed for verl reward fn: low-latency, robust parsing, safe fallback.
    """

    def __init__(
        self,
        model: str = "qwen-flash",
        api_key_env: str = "<YOUR_API_KEY>",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_sec: int = 60,
        max_retries: int = 2,
        backoff_sec: float = 0.8,
        enable_thinking: bool = False,
        thinking_budget: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_cache_size: int = 50000,
    ):
        api_key = "<YOUR_API_KEY>"
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Please set env var {api_key_env}."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.temperature = temperature
        self.top_p = top_p

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_fifo: list[str] = []
        self._max_cache_size = max_cache_size

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(key)

    def _cache_put(self, key: str, val: Dict[str, Any]) -> None:
        if key in self._cache:
            self._cache[key] = val
            return
        self._cache[key] = val
        self._cache_fifo.append(key)
        if len(self._cache_fifo) > self._max_cache_size:
            old = self._cache_fifo.pop(0)
            self._cache.pop(old, None)

    def judge(self, pred: str, gt: str) -> Dict[str, Any]:
        cache_key = _stable_hash(self.model, pred, gt)
        hit = self._cache_get(cache_key)
        if hit is not None:
            return hit

        prompt = TEXTUAL_WM_EVAL_PROMPT.format(PRED=pred, GT=gt)

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False,
                    extra_body=(
                        {"enable_thinking": True, "thinking_budget": int(self.thinking_budget)}
                        if self.enable_thinking
                        else {"enable_thinking": False}
                    ),
                    # Note: OpenAI python SDK doesn't universally expose request timeout in all versions;
                    # if your version supports it, you can pass `timeout=self.timeout_sec`.
                )
                content = completion.choices[0].message.content or ""
                obj = self._parse_judge_json(content)
                if obj is None:
                    # fallback: return all zeros if cannot parse
                    obj = {
                        "scores": {
                            "app_name": 0,
                            "user_action": 0,
                            "title_bar": 0,
                            "ribbon": 0,
                            "main_editing_area": 0,
                            "sidebar_pane": 0,
                            "navigation_area": 0,
                            "status_bar": 0,
                        },
                        "notes": {k: "Parse failed" for k in [
                            "app_name","user_action","title_bar","ribbon","main_editing_area",
                            "sidebar_pane","navigation_area","status_bar"
                        ]},
                    }
                self._cache_put(cache_key, obj)
                return obj
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.backoff_sec * (2 ** attempt))
                    continue
                break

        # Hard fallback on repeated failure
        fallback = {
            "scores": {
                "app_name": 0,
                "user_action": 0,
                "title_bar": 0,
                "ribbon": 0,
                "main_editing_area": 0,
                "sidebar_pane": 0,
                "navigation_area": 0,
                "status_bar": 0,
            },
            "notes": {k: f"Judge error: {type(last_err).__name__}" for k in [
                "app_name","user_action","title_bar","ribbon","main_editing_area",
                "sidebar_pane","navigation_area","status_bar"
            ]},
        }
        self._cache_put(cache_key, fallback)
        return fallback

    def _parse_judge_json(self, text: str) -> Optional[Dict[str, Any]]:
        cand = _extract_json_object(text)
        if not cand:
            return None
        # Try strict load; if fails, try light repair.
        for s in (cand, self._light_json_repair(cand)):
            try:
                obj = json.loads(s)
                if not isinstance(obj, dict) or "scores" not in obj:
                    continue
                # Normalize scores
                scores = obj.get("scores", {})
                if not isinstance(scores, dict):
                    continue
                for k in [
                    "app_name","user_action","title_bar","ribbon","main_editing_area",
                    "sidebar_pane","navigation_area","status_bar"
                ]:
                    scores[k] = _coerce_score(scores.get(k, 0))
                obj["scores"] = scores
                # notes optional; keep if exists
                return obj
            except Exception:
                continue
        return None

    @staticmethod
    def _light_json_repair(s: str) -> str:
        # Replace single quotes with double quotes (best effort)
        # and remove trailing commas.
        t = s.strip()
        t = re.sub(r"(?<!\\)'", '"', t)
        t = re.sub(r",\s*([}\]])", r"\1", t)
        return t


# -----------------------------
# 4) Reward shaping
# -----------------------------
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


# -----------------------------
# 5) verl reward entry
# -----------------------------
# Create one global judge instance (so cache is reused across calls in the same process)
_GLOBAL_QWEN_JUDGE: Optional[QwenJudge] = None


def _get_judge(**kwargs) -> QwenJudge:
    global _GLOBAL_QWEN_JUDGE
    if _GLOBAL_QWEN_JUDGE is None:
        _GLOBAL_QWEN_JUDGE = QwenJudge(**kwargs)
    return _GLOBAL_QWEN_JUDGE


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
    _ = _maybe_json_load(extra_info) or {}

    judge_model = kwargs.get("judge_model", "qwen-plus")
    enable_thinking = bool(kwargs.get("enable_thinking", False))
    thinking_budget = int(kwargs.get("thinking_budget", 0))
    base_url = kwargs.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    judge = _get_judge(
        model=judge_model,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
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

    reward = _weighted_score(scores, weights)

    # Small shaping to reduce rambling (which increases hallucination risk)
    if bool(kwargs.get("use_length_penalty", True)):
        reward -= _ratio_length_penalty_by_tokens(
            pred=pred,
            gt=gt,
            lower_ratio=float(kwargs.get("lower_ratio", 0.75)),
            upper_ratio=float(kwargs.get("upper_ratio", 1.25)),
            max_penalty=float(kwargs.get("max_penalty", 0.25)),
            encoding_name=str(kwargs.get("encoding_name", "cl100k_base")),
        )

    # Clamp
    if reward < 0.0:
        reward = 0.0
    if reward > 1.0:
        reward = 1.0
    return float(reward)


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




# -------------------------
# verl interface
# -------------------------
# score = self.compute_score(
#     data_source=data_source,
#     solution_str=response_str,
#     ground_truth=ground_truth,
#     extra_info=extra_info,
# )
# def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
#     """
#     目标：让 policy 输出的描述更贴合 GT next.png，并且描述“变化”正确，减少 hallucination。
#     需要 extra_info 里至少有:
#       - next_image
#       - action
#     推荐再加:
#       - prev_image (从 dataset 的 images[0] 或 extra_info 传进来)
#       - assistant_text (teacher)
#     """
#     import pdb;pdb.set_trace()
#     if data_source not in ["ui_world_model", "ui_world_model_rl"]:
#         return 0.0

#     pred = (solution_str or "").strip()
#     if not pred:
#         return 0.0

#     extra_info = _maybe_json_load(extra_info) or {}
#     action = extra_info.get("action") or {}
#     teacher = extra_info.get("assistant_text") or extra_info.get("reference_text") or ""

#     return 0
