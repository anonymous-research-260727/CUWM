
import json
import re
import time
import threading
from examples.ui_world_model.cloudgpt_aoai import get_chat_completion, encode_image, get_openai_client
from typing import Any, Dict, Optional
import hashlib
# 你已有的工具函数（这里假设已存在）
# - _stable_hash
# - _extract_json_object
# - _coerce_score
# - TEXTUAL_WM_EVAL_PROMPT

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


class GPTJudge:
    """
    Thread-safe GPT Judge wrapper with retry + cache.
    Uses cloudgpt_aoai.get_openai_client() for OpenAI-compatible GPT endpoints.
    """

    def __init__(
        self,
        model: str = "gpt-4o-20241120",
        timeout_sec: int = 60,
        max_retries: int = 4,
        backoff_sec: float = 0.8,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_cache_size: int = 50000,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model: GPT model name, e.g. "gpt-5.2-chat-20251211"
            timeout_sec: request timeout (if client supports it; cloudgpt client may ignore)
            max_retries: retry count on request failures
            backoff_sec: base backoff seconds
            temperature/top_p: decoding params
            max_cache_size: in-memory FIFO cache size
            request_kwargs: extra kwargs forwarded to client.chat.completions.create(...)
                           e.g. {"max_tokens": 512}
        """
        self.model = model
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.temperature = temperature
        self.top_p = top_p

        self.request_kwargs = request_kwargs or {}

        # cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_fifo: list[str] = []
        self._max_cache_size = max_cache_size
        self._cache_lock = threading.Lock()

        # per-thread client
        self._tls = threading.local()

    # -----------------------------
    # Thread-local client
    # -----------------------------
    def _get_client(self):
        c = getattr(self._tls, "client", None)
        if c is None:
            c = get_openai_client()
            self._tls.client = c
        return c

    # -----------------------------
    # Cache
    # -----------------------------
    def _cache_get(self, key: str):
        with self._cache_lock:
            return self._cache.get(key)

    def _cache_put(self, key: str, val: Dict[str, Any]):
        with self._cache_lock:
            if key in self._cache:
                self._cache[key] = val
                return
            self._cache[key] = val
            self._cache_fifo.append(key)
            if len(self._cache_fifo) > self._max_cache_size:
                old = self._cache_fifo.pop(0)
                self._cache.pop(old, None)

    # -----------------------------
    # Main
    # -----------------------------
    def judge(self, pred: str, gt: str):
        cache_key = _stable_hash(self.model, pred, gt)
        hit = self._cache_get(cache_key)
        if hit is not None:
            return hit

        prompt = TEXTUAL_WM_EVAL_PROMPT.format(PRED=pred, GT=gt)

        last_err = None
        last_raw = ""

        for attempt in range(self.max_retries + 1):
            try:
                client = self._get_client()

                # 调 GPT
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False,
                    **self.request_kwargs,
                )

                content = resp.choices[0].message.content or ""
                last_raw = content

                obj = self._parse_judge_json(content)
                if obj is None:
                    obj = self._make_parse_failed_payload(
                        cache_key=cache_key,
                        attempt=attempt,
                        raw_content=content,
                    )

                self._cache_put(cache_key, obj)
                return obj

            except Exception as e:
                last_err = e
                # 重试
                if attempt < self.max_retries:
                    time.sleep(self.backoff_sec * (2 ** attempt))
                    continue
                break

        # 最终 fallback
        fallback = self._make_error_payload(
            cache_key=cache_key,
            err=last_err,
            raw_content=last_raw,
        )
        self._cache_put(cache_key, fallback)
        return fallback

    # -----------------------------
    # JSON parsing
    # -----------------------------
    def _parse_judge_json(self, text: str):
        if not text:
            return None

        # 1) 去掉 markdown code fence（很多模型会输出 ```json ... ```）
        t = text.strip()
        t = t.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

        # 2) 抽取第一段 {...}
        cand = _extract_json_object(t)
        if not cand:
            return None

        # 3) strict load + repair load
        for s in (cand, self._light_json_repair(cand)):
            try:
                obj = json.loads(s)
                if not isinstance(obj, dict) or "scores" not in obj:
                    continue

                scores = obj.get("scores", {})
                if not isinstance(scores, dict):
                    continue

                for k in [
                    "app_name", "user_action", "title_bar", "ribbon", "main_editing_area",
                    "sidebar_pane", "navigation_area", "status_bar"
                ]:
                    scores[k] = _coerce_score(scores.get(k, 0))
                obj["scores"] = scores

                # notes 允许缺失
                if "notes" not in obj or not isinstance(obj.get("notes"), dict):
                    obj["notes"] = {k: "" for k in scores.keys()}

                return obj
            except Exception:
                continue

        return None

    @staticmethod
    def _light_json_repair(s: str):
        # best-effort: 单引号 -> 双引号，去 trailing comma
        t = s.strip()
        t = re.sub(r"(?<!\\)'", '"', t)
        t = re.sub(r",\s*([}\]])", r"\1", t)
        return t

    # -----------------------------
    # Payload makers
    # -----------------------------
    @staticmethod
    def _make_parse_failed_payload(cache_key: str, attempt: int, raw_content: str):
        keys = [
            "app_name", "user_action", "title_bar", "ribbon", "main_editing_area",
            "sidebar_pane", "navigation_area", "status_bar"
        ]
        return {
            "scores": {k: 0 for k in keys},
            "notes": {k: f"Parse failed (attempt={attempt})" for k in keys},
            "_debug": {
                "cache_key": cache_key,
                "model": "unknown_in_payload",  # 上层知道 model，这里不重复也行
                "attempt": attempt,
                "content_head": (raw_content or "")[:400],
            },
        }

    def _make_error_payload(self, cache_key: str, err: Exception, raw_content: str):
        keys = [
            "app_name", "user_action", "title_bar", "ribbon", "main_editing_area",
            "sidebar_pane", "navigation_area", "status_bar"
        ]
        return {
            "scores": {k: 0 for k in keys},
            "notes": {k: f"Judge error: {type(err).__name__ if err else 'Unknown'}" for k in keys},
            "_debug": {
                "cache_key": cache_key,
                "model": self.model,
                "error": repr(err),
                "content_head": (raw_content or "")[:400],
            },
        }
        
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