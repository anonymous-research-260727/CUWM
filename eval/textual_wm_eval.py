import os
import json
from tqdm import tqdm
from rouge import Rouge
from prompts.textual_wm_eval import TEXTUAL_WM_EVAL_PROMPT
from utils.cloudgpt_aoai import get_chat_completion, encode_image, get_openai_client
from concurrent.futures import ThreadPoolExecutor, as_completed


def llm_evaluate(gt, pred, client):
    prompt = TEXTUAL_WM_EVAL_PROMPT.format(GT=gt, PRED=pred)
    
    response = client.chat.completions.create(
        # model="gpt-4o-mini-20240718",
        model = "gpt-5.2-chat-20251211",
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
    
    eval_file = f"output/{model_key}-llm-eval.json"
    
   
    if not os.path.exists(eval_file):
        client = get_openai_client()
        
        def process_item(idx, item):
            gt = item["gt"]
            pred = item["pred"]
            eval_result = llm_evaluate(gt, pred, client)
            return idx, json.loads(eval_result)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_item, idx, item): idx for idx, item in enumerate(data)}
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx, eval_result = future.result()
                print(f"Processed item {idx}")
                data[idx]["eval"] = eval_result
                
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    
    eval_num = len(eval_data)
    eval_res = {"average": 0}
    for ele in eval_data:
        for key in ele["eval"]["scores"]:
            if key in eval_res:
                eval_res[key] += ele["eval"]["scores"][key] / eval_num
            else:
                eval_res[key] = ele["eval"]["scores"][key] / eval_num
            
            eval_res["average"] += ele["eval"]["scores"][key] / (eval_num * len(ele["eval"]["scores"]))
    
    scores = rouge.get_scores(hyps, refs, avg=True)
    eval_res["rouge-1"] = scores['rouge-1']['f']
    eval_res["rouge-2"] = scores['rouge-2']['f']
    eval_res["rouge-l"] = scores['rouge-l']['f']
    
    print("Evaluation Results:")
    print(json.dumps(eval_res, indent=2))
    
    with open(f"output/{model_key}-eval-summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_res, f, indent=2, ensure_ascii=False)    
    
    
    

if __name__ == "__main__":
    model_keys = [
        "gpt-5.2-chat-20251211",
        "checkpoint-450-merged",
        "base",
    ]
    for mk in model_keys:
        print(f"Evaluating model: {mk}")
        evaluate(mk)
    