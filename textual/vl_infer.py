import logging
import os
import json
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
from swift.llm import PtEngine, RequestConfig, InferRequest

logger = logging.getLogger(__name__)


def process_dataset(data):
    infer_requests = []
    for idx, ele in enumerate(data):
        assert ele["messages"][0]["role"] == "user"
        prompt = ele["messages"][0]["content"]
        images = ele["images"]
        infer_requests.append(
            InferRequest(
                messages=[{"role": "user", "content": prompt}],
                images=images,
            )
        )
    return infer_requests
    

if __name__ == "__main__":
    # model_key = "checkpoint-450"
    # model_key = "gpt-5.2-chat-20251211"
    model_key = "base"
    
    with open("data/test_data_1231_339.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        logger.info("Loaded %d samples for inference.", len(data))
        
    model_dict = {
        "base": os.environ.get("MODEL_BASE", "models/Qwen") + "/Qwen2.5-VL-7B-Instruct",
        "checkpoint-450-merged": 'output/prompt_1228/v3-20251231-143109/checkpoint-450-merged',
    }
        
    if model_key in ["base", "checkpoint-450-merged"]:
        model = model_dict[model_key]

        # Load the inference engine
        engine = PtEngine(model, max_batch_size=32)
        request_config = RequestConfig(max_tokens=2048, temperature=0)
        infer_requests = process_dataset(data)
        logger.info("Total inference requests: %d", len(infer_requests))
        
        resp_list = engine.infer(infer_requests, request_config)
        
        for idx, ele in enumerate(data):
            assert ele["messages"][1]["role"] == "assistant"
            data[idx]["gt"] = ele["messages"][1]["content"]
            data[idx]["pred"] = resp_list[idx].choices[0].message.content.strip()
            
    elif model_key == "gpt-5.2-chat-20251211":
        from utils.cloudgpt_aoai import get_openai_client
        from scripts.generate_text_pred_prompt import generate_nl_description
        from concurrent.futures import ThreadPoolExecutor, as_completed
        client = get_openai_client()
        
        def process_single_item(idx, ele):
            folder = os.path.dirname(ele["images"][0])
            content, prompt = generate_nl_description(
            folder,
            client=client,
            llm_model="gpt-5.2-chat-20251211",
            )
            # print("prompt:", prompt)
            # assert 0
            return idx, content
        
        results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_single_item, idx, ele): idx for idx, ele in enumerate(data)}
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx, content = future.result()
                results[idx] = content
        
        for idx, ele in enumerate(data):
            data[idx]["gt"] = ele["messages"][1]["content"]
            assert ele["messages"][1]["role"] == "assistant"
            data[idx]["pred"] = results[idx]
            
    else:
        raise ValueError(f"Unknown model key: {model_key}")
        
    with open(f"output/{model_key}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Inference results saved.")
        
    