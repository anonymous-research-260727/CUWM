import os
import json
from tqdm import tqdm
from rouge import Rouge
# from prompts.textual_wm_eval import TEXTUAL_WM_EVAL_PROMPT
from cloudgpt_aoai import get_chat_completion, encode_image, get_openai_client
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys; sys.setrecursionlimit(10000)


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

# DEFAULT_WEIGHTS = {
#   "app_name": 0.05,
#   "user_action": 0.15,
#   "title_bar": 0.10,
#   "ribbon": 0.20,
#   "main_editing_area": 0.30,
#   "sidebar_pane": 0.10,
#   "navigation_area": 0.05,
#   "status_bar": 0.05,
# }

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


def llm_evaluate(gt, pred, client):
    prompt = TEXTUAL_WM_EVAL_PROMPT.format(GT=gt, PRED=pred)
    
    response = client.chat.completions.create(
        # model="gpt-4o-mini-20240718",
        # model = "gpt-5.2-chat-20251211",
        model = "gpt-5.2-20251211",
        messages=[{"role": "user", "content": prompt}],
        # temperature=0.7,
        # max_tokens=100,
        # top_p=0.95,
        # frequency_penalty=0,
        # presence_penalty=0,
    )
    # response = get_chat_completion(
    #     model="gpt-5.2-chat-20251211",
    #     messages=messages,
    # )
    content = response.choices[0].message.content
    content = content.replace("```json", "").replace("```", "").strip()
    return content

def evaluate(model_key: str):
    rouge = Rouge()
    
    with open(f"output/{model_key}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    refs = [item["gt"] for item in data]
    hyps = [item["pred"] for item in data]
    
    eval_file = f"output/{model_key}-llm-eval-in-gpt-5.2-20251211.json"
    
   
    if not os.path.exists(eval_file):
        client = get_openai_client()
        
        def process_item(idx, item):
            gt = item["gt"]
            pred = item["pred"]
            eval_result = llm_evaluate(gt, pred, client)
            return idx, json.loads(eval_result)
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(process_item, idx, item): idx for idx, item in enumerate(data)}
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx, eval_result = future.result()
                print(f"Processed item {idx}")
                data[idx]["eval"] = eval_result
                
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    
    # eval_num = len(eval_data)
    # eval_res = {"average": 0}
    # for ele in eval_data:
    #     for key in ele["eval"]["scores"]:
    #         if key in eval_res:
    #             eval_res[key] += ele["eval"]["scores"][key] / eval_num
    #         else:
    #             eval_res[key] = ele["eval"]["scores"][key] / eval_num
            
    #         eval_res["average"] += ele["eval"]["scores"][key] / (eval_num * len(ele["eval"]["scores"]))
    
    eval_num = len(eval_data)
    eval_res = {
        "average": 0.0,           # 你原来的等权平均
        "weighted_average": 0.0,  # ✅ 新增：加权总分
    }

    num_keys = None  # 用于等权 average 的分母

    for ele in eval_data:
        scores_dict = ele["eval"]["scores"]

        # 记录每个子项的平均（保持你原逻辑）
        for key in scores_dict:
            if key in eval_res:
                eval_res[key] += float(scores_dict[key]) / eval_num
            else:
                eval_res[key] = float(scores_dict[key]) / eval_num

        # 你原来的等权 average（按样本平均）
        if num_keys is None:
            num_keys = len(scores_dict)
        eval_res["average"] += sum(float(v) for v in scores_dict.values()) / (eval_num * num_keys)

        # ✅ 新增：加权 average（按样本平均）
        eval_res["weighted_average"] += weighted_score(scores_dict, DEFAULT_WEIGHTS) / eval_num

    
    scores = rouge.get_scores(hyps, refs, avg=True)
    eval_res["rouge-1"] = scores['rouge-1']['f']
    eval_res["rouge-2"] = scores['rouge-2']['f']
    eval_res["rouge-l"] = scores['rouge-l']['f']
    
    eval_res_4 = round_floats(eval_res, ndigits=4)

    print("Evaluation Results:")
    print(json.dumps(eval_res_4, indent=2, ensure_ascii=False))

    with open(f"output/{model_key}-eval-summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_res_4, f, indent=2, ensure_ascii=False)
  
    
    
    

if __name__ == "__main__":
    model_keys = [
        "base",
        "checkpoint-450-merged",
        "checkpoint-450-grpo",
        "checkpoint-450-grpo-ckpt250",
        "checkpoint-450-grpo-pairwise",
        "gpt-5.2-chat-20251211"
    ]
    for mk in model_keys:
        print(f"Evaluating model: {mk}")
        evaluate(mk)
    