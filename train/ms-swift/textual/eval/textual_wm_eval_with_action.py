# eval_with_action_coherence.py
import os
import json
import re
import glob
import base64
from typing import Dict, Optional, Any

from tqdm import tqdm
from rouge import Rouge
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

from cloudgpt_aoai import get_openai_client
from openai import OpenAI

from action_evaluation_custom import ActionEvaluationA11y

sys.setrecursionlimit(10000)

# =========================================================
# 0) Textual WM Judge Prompt (你原来的)
# =========================================================
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

DEFAULT_WEIGHTS = {
    "app_name": 0.10,
    "user_action": 0.175,
    "title_bar": 0.125,
    "ribbon": 0.1375,
    "main_editing_area": 0.1875,
    "sidebar_pane": 0.10,
    "navigation_area": 0.075,
    "status_bar": 0.10,
}


def weighted_score(scores: dict, weights: dict) -> float:
    num = 0.0
    den = 0.0
    for k, w in weights.items():
        w = float(w)
        if w <= 0:
            continue
        if k not in scores:
            continue
        num += float(scores[k]) * w
        den += w
    return num / max(1e-8, den)


def round_floats(obj, ndigits=4):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def llm_evaluate(gt: str, pred: str, client) -> dict:
    prompt = TEXTUAL_WM_EVAL_PROMPT.format(GT=gt, PRED=pred)
    response = client.chat.completions.create(
        model="gpt-5.2-20251211",
        messages=[{"role": "user", "content": prompt}],
    )
    content = (response.choices[0].message.content or "").replace("```json", "").replace("```", "").strip()
    return json.loads(content)


# =========================================================
# 1) Action Coherence (从你 reward 里抽出来，做 eval 用)
# =========================================================
actEval = ActionEvaluationA11y()

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

Your response MUST be a valid JSON array with exactly one element and no additional text.
""".strip()

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


def _action_client_from_env(model) -> OpenAI:
    """
    建议用环境变量提供 action 模型的 endpoint：
      export ACTION_BASE_URL="https://api.xxx/v1"
      export ACTION_API_KEY="sk-xxxx"
    """
    if model == "qwen3-vl-flash":
        base_url = "https://api.vectorengine.ai/v1"
        api_key = "<YOUR_API_KEY>"
    if model == "qwen3-vl-8b-instruct":
        base_url = "https://api.vectorengine.ai/v1"
        api_key = "<YOUR_API_KEY>"
    if model == "gpt-4.1-mini":
        base_url = "https://api.vectorengine.ai/v1"
        api_key = "<YOUR_API_KEY>"
    if model == "gpt-4o":
        base_url = "https://api.vectorengine.ai/v1"
        api_key = "<YOUR_API_KEY>"
    if model == "gemini-2.0-flash":
        base_url = "https://api.vectorengine.ai/v1"
        api_key = "<YOUR_API_KEY>"
    if not base_url or not api_key:
        raise RuntimeError(
            "Missing ACTION_BASE_URL / ACTION_API_KEY. "
            "Please set env vars for action model endpoint."
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def _call_action_llm(messages, model: str, temperature: float = 0.0, **kwargs) -> str:
    """
    OpenAI 兼容接口调用：
      - 可传 action_client=OpenAI(...)
      - 或用环境变量 ACTION_BASE_URL / ACTION_API_KEY
    """
    client = kwargs.get("action_client")
    if client is None:
        client = _action_client_from_env(model=model)

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
        a11y=json.dumps(a11y, indent=2, ensure_ascii=False),
        actions=supported_actions,
    )
    return ACTION_PREDICTION_A11Y_SYS_PROMPT_GPT + "\n\n" + usr_prompt


def _parse_action_tool_call(text: str) -> Optional[dict]:
    """
    兼容两种返回：
      1) [ {"tool_call": {...}} ]
      2) {"tool_call": {...}}
      3) 直接 {...}（当模型没包 tool_call）
    """
    if not text:
        return None
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(text)
    except Exception:
        return None

    if isinstance(obj, list) and len(obj) >= 1:
        first = obj[0]
        if isinstance(first, dict):
            tc = first.get("tool_call", first)
            return tc if isinstance(tc, dict) else None
        return None

    if isinstance(obj, dict):
        tc = obj.get("tool_call", obj)
        return tc if isinstance(tc, dict) else None

    return None


def _get_action(
    img_path: Optional[str],
    screen_description: Optional[str],
    screen_description_path: Optional[str],
    a11y_path: str,
    instruction_path: str,
    **kwargs,
) -> Optional[dict]:
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

    action_model = kwargs.get("action_model", "qwen3-vl-flash")
    max_retry = int(kwargs.get("action_max_retry", 3))

    for _ in range(max_retry):
        try:
            resp = _call_action_llm(messages, model=action_model, temperature=0.0, **kwargs)
            tc = _parse_action_tool_call(resp)
            if tc is not None:
                return tc
        except Exception:
            continue
    return None


def _action_coherence_eval(
    textual_wm_response: str,
    img_path: str,
    gt_description_path: str,
    a11y_path: str,
    instruction_path: str,
    **kwargs,
) -> dict:
    """
    输出：
      {
        "action_match": float(0~1),
        "function_match": bool,
        "status_match": bool,
        "args_match": bool,
        "action_pred": {...} | None,
        "action_gt": {...} | None
      }
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
        return {
            "action_match": 0.0,
            "function_match": False,
            "status_match": False,
            "args_match": False,
            "action_pred": action_pred,
            "action_gt": action_gt,
        }

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

    return {
        "action_match": float(score),
        "function_match": function_match,
        "status_match": status_match,
        "args_match": args_match,
        "action_pred": action_pred,
        "action_gt": action_gt,
    }


def _infer_pair_files(pair_path: str) -> Dict[str, str]:
    """
    从 pair 目录里自动找：
      - next.png / next.*
      - a11y.json：优先取“下一个 pair”的 a11y（如 pair_01 -> pair_02），若不存在则取当前 pair 的
      - request.txt / *request*.txt
      - prompt_nl_gt.txt（优先）/ gt*.txt
    """
    pair_path = pair_path or "."

    # next
    nxt = os.path.join(pair_path, "next.png")
    if not os.path.exists(nxt):
        cands = glob.glob(os.path.join(pair_path, "next.*"))
        if cands:
            nxt = cands[0]

    def _next_pair_dir(cur_pair_dir: str) -> Optional[str]:
        base = os.path.basename(os.path.normpath(cur_pair_dir))
        m = re.match(r"^(pair_)(\d+)$", base)
        if not m:
            return None
        prefix, num = m.group(1), m.group(2)
        width = len(num)
        nxt_num = str(int(num) + 1).zfill(width)
        return os.path.join(os.path.dirname(os.path.normpath(cur_pair_dir)), f"{prefix}{nxt_num}")

    # a11y (prefer next pair)
    a11y = None
    npair = _next_pair_dir(pair_path)
    if npair and os.path.isdir(npair):
        cand = os.path.join(npair, "a11y.json")
        if os.path.exists(cand):
            a11y = cand
        else:
            cands = glob.glob(os.path.join(npair, "*a11y*.json"))
            if cands:
                a11y = cands[0]
    # if a11y is None:
    #     cand = os.path.join(pair_path, "a11y.json")
    #     if os.path.exists(cand):
    #         a11y = cand
    #     else:
    #         cands = glob.glob(os.path.join(pair_path, "*a11y*.json"))
    #         a11y = cands[0] if cands else cand

    # request
    req = os.path.join(pair_path, "request.txt")
    if not os.path.exists(req):
        cands = glob.glob(os.path.join(pair_path, "*request*.txt"))
        if cands:
            req = cands[0]

    # gt description
    gt_cands = glob.glob(os.path.join(pair_path, "prompt_nl_gt.txt"))
    if not gt_cands:
        gt_cands = glob.glob(os.path.join(pair_path, "gt*.txt"))
    gt = gt_cands[0] if gt_cands else ""

    return {"pair_path": pair_path, "next": nxt, "a11y": a11y, "request": req, "gt": gt}


def _get_pair_path_from_item(item: dict):
    """
    尽量兼容你可能存的字段：
      - item["pair_path"]
      - item["extra_info"]["pair_path"]
      - item["meta"]["pair_path"]
    """
    if not isinstance(item, dict):
        return None
    if isinstance(item, dict) and isinstance(item.get("images"), list):
        return os.path.dirname(item["images"][0])
    return None


# =========================================================
# 2) Main Eval
# =========================================================
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from rouge import Rouge

def evaluate(model_key: str):
    rouge = Rouge()

    src_file = f"output_action/{model_key}.json"
    with open(src_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    refs = [item["gt"] for item in data]
    hyps = [item["pred"] for item in data]

    # -----------------------------
    # 1) Textual LLM judge (detail + summary)
    # -----------------------------
    llm_detail_file = f"output_action/{model_key}-llm-eval-in-gpt-5.2-20251211.json"
    llm_summary_file = f"output_action/{model_key}-eval-summary.json"

    if not os.path.exists(llm_detail_file):
        client = get_openai_client()

        def process_llm_item(idx: int, item: dict):
            gt = item["gt"]
            pred = item["pred"]
            eval_result = llm_evaluate(gt, pred, client)
            return idx, eval_result

        llm_data = [dict(x) for x in data]  # 拷贝一份，避免污染原 data
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(process_llm_item, idx, item): idx
                for idx, item in enumerate(llm_data)
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx, eval_result = future.result()
                llm_data[idx]["eval"] = eval_result
                # 不再写 action_eval 到这个文件
                # print(f"Processed LLM item {idx}")

        with open(llm_detail_file, "w", encoding="utf-8") as f:
            json.dump(llm_data, f, indent=2, ensure_ascii=False)

    with open(llm_detail_file, "r", encoding="utf-8") as f:
        llm_eval_data = json.load(f)

    eval_num = len(llm_eval_data)
    eval_res = {
        "average": 0.0,
        "weighted_average": 0.0,
    }

    num_keys = None
    for ele in llm_eval_data:
        scores_dict = ele["eval"]["scores"]

        for key in scores_dict:
            if key in eval_res:
                eval_res[key] += float(scores_dict[key]) / eval_num
            else:
                eval_res[key] = float(scores_dict[key]) / eval_num

        if num_keys is None:
            num_keys = len(scores_dict)

        eval_res["average"] += sum(float(v) for v in scores_dict.values()) / (eval_num * num_keys)
        eval_res["weighted_average"] += weighted_score(scores_dict, DEFAULT_WEIGHTS) / eval_num

    # rouge 基于原始 pred/gt（不依赖 llm 结果）
    scores = rouge.get_scores(hyps, refs, avg=True)
    eval_res["rouge-1"] = scores["rouge-1"]["f"]
    eval_res["rouge-2"] = scores["rouge-2"]["f"]
    eval_res["rouge-l"] = scores["rouge-l"]["f"]

    eval_res_4 = round_floats(eval_res, ndigits=4)

    print("Textual LLM Judge + Rouge Summary:")
    print(json.dumps(eval_res_4, indent=2, ensure_ascii=False))

    # 你要求 textual summary 仍沿用这个文件名
    with open(llm_summary_file, "w", encoding="utf-8") as f:
        json.dump(eval_res_4, f, indent=2, ensure_ascii=False)

    # -----------------------------
    # 2) Action coherence (detail + summary) - separated cache
    # -----------------------------
    # ACTION_MODEL = "gemini-2.0-flash"
    # ACTION_MODEL = "gpt-4.1-mini"
    # ACTION_MODEL = "gpt-4o"
    ACTION_MODEL = "qwen3-vl-8b-instruct"
    action_detail_file = f"output_action/{model_key}_{ACTION_MODEL}_action-eval.json"
    action_summary_file = f"output_action/{model_key}_{ACTION_MODEL}_action-summary.json"

    # enable_action_eval：如果你不想跑动作一致性，把 ENABLE_ACTION_EVAL=0
    enable_action_eval = 1

    if enable_action_eval:
        if not os.path.exists(action_detail_file):
            action_client = None
            try:
                action_client = _action_client_from_env(model=ACTION_MODEL)
            except Exception as e:
                print(f"[WARN] Action eval disabled because env not set: {e}")
                enable_action_eval = False

            if enable_action_eval:
                def process_action_item(idx: int, item: dict):
                    pred = item["pred"]
                    action_eval = None
                    # import pdb;pdb.set_trace()
                    pair_path = _get_pair_path_from_item(item)
                    if pair_path is None:
                        print(f"[WARN] item {idx} missing pair_path info for action eval.")
                    if pair_path:
                        files = _infer_pair_files(pair_path)
                        # 必要文件存在才评
                        if (
                            files.get("gt")
                            and os.path.exists(files["next"])
                            and files["a11y"] is not None
                            and os.path.exists(files["a11y"])
                            and os.path.exists(files["request"])
                            and os.path.exists(files["gt"])
                        ):
                            action_eval = _action_coherence_eval(
                                textual_wm_response=pred,
                                img_path=files["next"],
                                gt_description_path=files["gt"],
                                a11y_path=files["a11y"],
                                instruction_path=files["request"],
                                action_client=action_client,
                                action_model=ACTION_MODEL,
                                action_max_retry=int(os.getenv("ACTION_MAX_RETRY", "3")),
                            )
                            action_eval["files"] = files
                        else:
                            action_eval = {
                                "action_match": 0.0,
                                "function_match": False,
                                "status_match": False,
                                "args_match": False,
                                "action_pred": None,
                                "action_gt": None,
                                "files": files,
                                "error": "missing required files",
                            }
                    else:
                        action_eval = {
                            "action_match": None,
                            "error": "pair_path not found in item (need item['pair_path'] or item['extra_info']['pair_path'])",
                        }
                    # if idx % 100 == 0:
                    #     print(f"Processed Action item {idx}")
                    #     print(json.dumps(action_eval, indent=2, ensure_ascii=False))
                    return idx, action_eval

                # 保存为“仅 action”的结构，避免跟 llm_detail_file 混在一起
                action_results = [None] * len(data)
                with ThreadPoolExecutor(max_workers=24) as executor:
                    futures = {
                        executor.submit(process_action_item, idx, item): idx
                        for idx, item in enumerate(data)
                    }
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        idx, action_eval = future.result()
                        action_results[idx] = action_eval
                        # print(f"Processed Action item {idx}")

                action_payload = {
                    "model_key": model_key,
                    "source": src_file,
                    "num_items": len(data),
                    "action_results": action_results,
                }
                with open(action_detail_file, "w", encoding="utf-8") as f:
                    json.dump(action_payload, f, indent=2, ensure_ascii=False)

        # 复用 action_detail_file 进行汇总
        if os.path.exists(action_detail_file):
            with open(action_detail_file, "r", encoding="utf-8") as f:
                action_payload = json.load(f)

            action_results = action_payload.get("action_results", [])
            action_match_sum = 0.0
            action_match_cnt = 0

            valid_cnt = 0
            err_cnt = 0

            for ae in action_results:
                if not isinstance(ae, dict):
                    continue

                # 有 error 的直接剔除
                if ae.get("error"):
                    err_cnt += 1
                    continue

                am = ae.get("action_match", None)

                # action_match 不是数字的也剔除（例如 None）
                if not isinstance(am, (int, float)):
                    err_cnt += 1
                    continue

                action_match_sum += float(am)
                action_match_cnt += 1
                valid_cnt += 1

            action_summary = {
                "action_match": (action_match_sum / action_match_cnt) if action_match_cnt > 0 else None,
                "action_match_num": action_match_cnt,          # 有效样本数（用于均值）
                "num_items": len(action_results),              # 总样本数
                "num_valid_items": valid_cnt,                  # 新增：有效样本数
                "num_error_items": err_cnt,                    # 新增：被剔除数
            }

            action_summary_4 = round_floats(action_summary, ndigits=4)

            print("\nAction Coherence Summary:")
            print(json.dumps(action_summary_4, indent=2, ensure_ascii=False))

            with open(action_summary_file, "w", encoding="utf-8") as f:
                json.dump(action_summary_4, f, indent=2, ensure_ascii=False)

    return eval_res_4



if __name__ == "__main__":
    model_keys = [
        "base",
        "checkpoint-450-merged",
        # "checkpoint-450-grpo",
        # "checkpoint-450-grpo-ckpt250",
        # "checkpoint-450-grpo-pairwise",
        # "grpo-action-ckpt-100",
        # "grpo-action-next-ckpt-100",
        "grpo-action-next-ckpt-150",
        "grpo-action-next-ckpt-600",
        # "gpt-5.2-chat-20251211",
    ]
    for mk in model_keys:
        print(f"Evaluating model: {mk}")
        evaluate(mk)
