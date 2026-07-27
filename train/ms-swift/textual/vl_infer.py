import os
import json
from tqdm import tqdm
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

from swift.llm import RequestConfig, InferRequest
from swift.plugin import InferStats


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


def infer_batch(engine, infer_requests, max_tokens=1024):
    """按参考代码风格封装 infer_batch（带 metric）"""
    # request_config = RequestConfig(max_tokens=max_tokens, temperature=0, top_p=1.0, repetition_penalty=1.0)
    request_config = RequestConfig(max_tokens=max_tokens, temperature=0)
    print("request_config:", request_config)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
    # 打印一个样例
    if len(infer_requests) > 0:
        query0 = infer_requests[0].messages[0]['content']
        print(f'query0: {query0[:200]}')  # 避免太长
        print(f'response0: {resp_list[0].choices[0].message.content[:200]}')
    try:
        print(f'metric: {metric.compute()}')
    except Exception as e:
        print(f'[WARN] metric.compute() failed: {e}')
    return resp_list


if __name__ == "__main__":
    model_key = "sft-ckpt-45"
    # model_key = "sft-ckpt-450"
    # model_key = "gpt-5.2-chat-20251211"
    # model_key = "checkpoint-450-grpo"
    # model_key = "checkpoint-450-grpo-pairwise"
    # model_key = "checkpoint-450-grpo-ckpt250"
    # model_key = "grpo-action-ckpt-100"
    # model_key = "grpo-action-next-ckpt-100"
    # model_key = "grpo-action-next-ckpt-150"
    # model_key = "grpo-action-next-ckpt-600"
    # model_key = "base"

    # ✅ 新增：选择推理后端（不改变整体流程，只换 engine）
    infer_backend = "vllm"  # "pt" 或 "vllm"

    with open("data/test_data_1231_339.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"✅ Loaded {len(data)} samples for inference.")
    # Normalize image path prefixes (replace old NVMe prefix with workspace path)
    old_prefix = "/path/to/cuwm"
    new_prefix = os.environ.get("CUWM_ROOT", "/path/to/cuwm")
    for item in data:
        if isinstance(item, dict) and "images" in item and item["images"]:
            item["images"] = [p.replace(old_prefix, new_prefix, 1) if isinstance(p, str) and p.startswith(old_prefix) else p for p in item["images"]]

    model_dict = {
        "base": "/path/to/cuwm/model/Qwen/Qwen2.5-VL-7B-Instruct",
        "sft-ckpt-45": os.path.join(new_prefix, "models/checkpoint-45-merged"),
        "sft-ckpt-450": "output/prompt_1228/v3-20251231-143109/checkpoint-450-merged",
        "checkpoint-450-grpo": "/path/to/cuwm/project/rl-0.6.0/verl/checkpoints/verl_grpo_officeWM3k/qwen2_5_vl_7b_officeWM/global_step_200/actor/huggingface",
        "checkpoint-450-grpo-pairwise": "/path/to/cuwm/project/rl-0.6.0/verl/checkpoints/verl_grpo_officeWM1k/qwen2_5_vl_7b_officeWM/global_step_200/actor/huggingface",
        "checkpoint-450-grpo-ckpt250": "/path/to/cuwm/project/rl-0.6.0/verl/checkpoints/verl_grpo_officeWM3k/qwen2_5_vl_7b_officeWM/global_step_250/actor/huggingface",
        # "grpo-action-ckpt-100": "/path/to/cuwm/project/rl-0.6.0/verl/checkpoints/verl_grpo_officeWM3k_action/qwen2_5_vl_7b_officeWM/global_step_100/actor/huggingface/",
        "grpo-action-next-ckpt-100": "/path/to/cuwm/project/rl-0.6.0/verl/checkpoints/verl_grpo_officeWM3k_action_next/qwen2_5_vl_7b_officeWM/global_step_100/actor/huggingface",
        "grpo-action-next-ckpt-150": "/path/to/cuwm/project/rl-0.6.0/verl/checkpoints/verl_grpo_officeWM3k_action_next/qwen2_5_vl_7b_officeWM/global_step_150/actor/huggingface",
        "grpo-action-next-ckpt-600": "/path/to/cuwm/project/rl-0.6.0/verl/checkpoints/verl_grpo_officeWM3k_action_next/qwen2_5_vl_7b_officeWM/global_step_600/actor/huggingface",
    }

    if model_key in ["base", "sft-ckpt-45", "sft-ckpt-450", "checkpoint-450-grpo", \
        "checkpoint-450-grpo-pairwise", "checkpoint-450-grpo-ckpt250", "grpo-action-ckpt-100", \
            "grpo-action-next-ckpt-100", "grpo-action-next-ckpt-150", "grpo-action-next-ckpt-600"]:
        model = model_dict[model_key]

        # ✅ 按参考代码：根据后端创建 engine
        if infer_backend == "pt":
            from swift.llm import PtEngine
            engine = PtEngine(model, max_batch_size=32)
        elif infer_backend == "vllm":
            from swift.llm import VllmEngine
            # 注意：max_model_len 至少要覆盖 prompt+response，建议 >= 4096+2048
            engine = VllmEngine(
                model,
                model_type="qwen2_5_vl",
                tensor_parallel_size=2,
                max_model_len=5120,
                gpu_memory_utilization=0.16,
                enforce_eager=True,
                torch_dtype=torch.bfloat16,
                max_num_seqs=8,
            )
        else:
            raise ValueError(f"Unknown infer_backend: {infer_backend}")

        infer_requests = process_dataset(data)
        print(f"Total inference requests: {len(infer_requests)}")

        # ✅ 按参考代码风格 infer_batch + metric
        resp_list = infer_batch(engine, infer_requests, max_tokens=1024)

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
            return idx, content

        results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_single_item, idx, ele): idx for idx, ele in enumerate(data)}
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx, content = future.result()
                results[idx] = content

        for idx, ele in enumerate(data):
            assert ele["messages"][1]["role"] == "assistant"
            data[idx]["gt"] = ele["messages"][1]["content"]
            data[idx]["pred"] = results[idx]

    else:
        raise ValueError(f"Unknown model key: {model_key}")

    out_path = f"output/{model_key}_{infer_backend}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Inference results saved to {out_path}.")
