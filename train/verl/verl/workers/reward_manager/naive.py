# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from collections import defaultdict
# from typing import Any

# import torch

# from verl import DataProto
# from verl.utils.reward_score import default_compute_score
# from verl.workers.reward_manager import register
# from verl.workers.reward_manager.abstract import AbstractRewardManager


# @register("naive")
# class NaiveRewardManager(AbstractRewardManager):
#     """The reward manager."""

#     def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
#         """
#         Initialize the NaiveRewardManager instance.

#         Args:
#             tokenizer: The tokenizer used to decode token IDs into text.
#             num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
#             compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
#             reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
#                 "data_source".
#         """
#         self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
#         self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
#         self.compute_score = compute_score or default_compute_score
#         self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

#     def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
#         """We will expand this function gradually based on the available datasets"""

#         # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
#         if "rm_scores" in data.batch.keys():
#             if return_dict:
#                 reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
#                 reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
#                 return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
#             else:
#                 return data.batch["rm_scores"]

#         reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
#         reward_extra_info = defaultdict(list)

#         already_print_data_sources = {}
#         print("len(data): ", len(data))
#         for i in range(len(data)):
#             data_item = data[i]  # DataProtoItem

#             prompt_ids = data_item.batch["prompts"]

#             prompt_length = prompt_ids.shape[-1]

#             valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
#             valid_prompt_ids = prompt_ids[-valid_prompt_length:]

#             response_ids = data_item.batch["responses"]
#             valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
#             valid_response_ids = response_ids[:valid_response_length]

#             # decode
#             prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
#             response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

#             ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
#             data_source = data_item.non_tensor_batch[self.reward_fn_key]
#             extra_info = data_item.non_tensor_batch.get("extra_info", {})
#             num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
#             rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
#             extra_info["num_turns"] = num_turns
#             extra_info["rollout_reward_scores"] = rollout_reward_scores

#             score = self.compute_score(
#                 data_source=data_source,
#                 solution_str=response_str,
#                 ground_truth=ground_truth,
#                 extra_info=extra_info,
#             )

#             if isinstance(score, dict):
#                 reward = score["score"]
#                 # Store the information including original reward
#                 for key, value in score.items():
#                     reward_extra_info[key].append(value)
#             else:
#                 reward = score

#             reward_tensor[i, valid_response_length - 1] = reward

#             if data_source not in already_print_data_sources:
#                 already_print_data_sources[data_source] = 0

#             if already_print_data_sources[data_source] < self.num_examine:
#                 already_print_data_sources[data_source] += 1
#                 print("[prompt]", prompt_str)
#                 print("[response]", response_str)
#                 print("[ground_truth]", ground_truth)
#                 if isinstance(score, dict):
#                     for key, value in score.items():
#                         print(f"[{key}]", value)
#                 else:
#                     print("[score]", score)
#         # import pdb;pdb.set_trace()
#         if return_dict:
#             return {
#                 "reward_tensor": reward_tensor,
#                 "reward_extra_info": reward_extra_info,
#             }
#         else:
#             return reward_tensor

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from collections import defaultdict
import os
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import json

def _as_float(x):
    # wandb/metric reduce 期望是数值；非数值就跳过
    try:
        # bool -> 0/1
        if isinstance(x, bool):
            return float(int(x))
        return float(x)
    except Exception:
        return None
    
def _borda_order(idxs, score_map, tie_eps=1e-6):
    """
    Pairwise/Borda: for each candidate, count wins vs others.
    Return: order (best->worst), borda_scores dict
    """
    borda = {i: 0.0 for i in idxs}
    for a in idxs:
        sa = score_map[a]
        for b in idxs:
            if a == b:
                continue
            sb = score_map[b]
            if sa > sb + tie_eps:
                borda[a] += 1.0
            elif sb > sa + tie_eps:
                borda[a] += 0.0
            else:
                borda[a] += 0.5
    order = sorted(idxs, key=lambda i: borda[i], reverse=True)
    return order, borda


def _rank_to_reward(rank, n):
    """
    Map rank -> [0,1], rank=0 best.
    Linear mapping: best=1, worst=0
    """
    if n <= 1:
        return 0.0
    return 1.0 - (rank / (n - 1))

def _is_train_phase(data: DataProto) -> bool:
    mi = getattr(data, "meta_info", None) or {}
    # 常见字段名：mode / split / stage / is_train
    if isinstance(mi, dict):
        if "is_train" in mi:
            return bool(mi["is_train"])
        if "mode" in mi:
            return str(mi["mode"]).lower() in ("train", "training")
        if "split" in mi:
            return str(mi["split"]).lower() in ("train",)
        if "stage" in mi:
            return str(mi["stage"]).lower() in ("train", "training")
        if "validate" in mi:
            return not bool(mi["validate"])
    return True  # 如果没有任何标记，默认当作 train（更安全：避免训练时意外关掉）

@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

        # ✅ 并发控制：可通过环境变量调参
        self.max_workers = int(os.getenv("REWARD_MAX_WORKERS", "24"))
        self.max_inflight = int(os.getenv("REWARD_MAX_INFLIGHT", str(self.max_workers)))

        # ✅ 打印控制
        self.log_fail_limit = int(os.getenv("REWARD_LOG_FAIL_LIMIT", "10"))   # 每次 __call__ 最多打印多少条失败
        self.log_ok_samples = int(os.getenv("REWARD_LOG_OK_SAMPLES", "3"))    # 每次 __call__ 打印多少条成功样例耗时
        self.print_every_call_summary = int(os.getenv("REWARD_PRINT_SUMMARY", "1"))  # 1=每次打印汇总

        # ✅ pairwise 开关/参数（默认开）
        self.enable_pairwise = bool(int(os.getenv("REWARD_ENABLE_PAIRWISE", "1")))
        self.pairwise_tie_eps = float(os.getenv("PAIRWISE_TIE_EPS", "1e-6"))
        # 期望的每组候选数（rollout.n），用于 debug
        self.expected_group_size = int(os.getenv("ROLLOUT_N", "5"))
        
        self.pairwise_print_groups = int(os.getenv("PAIRWISE_PRINT_GROUPS", "1"))  # 每次 call 最多打印多少个 uid 组
        self.pairwise_print_topk = int(os.getenv("PAIRWISE_PRINT_TOPK", "5"))      # 每组最多打印多少条候选

        
    def __call__(self, data: DataProto, return_dict: bool = False):
        # Fast path: already have rm score
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        n = len(data)
        print("len(data): ", n)

        # ----------------------------
        # 1) 主线程准备输入（避免多线程反复访问 DataProto）
        # ----------------------------
        prepared = []
        for i in range(n):
            item = data[i]
            
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = item.batch["responses"]
            valid_response_length = item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # ✅ 只 decode response；prompt 仅在打印时 decode
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = item.non_tensor_batch[self.reward_fn_key]
            extra_info = item.non_tensor_batch.get("extra_info", {})
            num_turns = item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = item.non_tensor_batch.get("reward_scores", {})

            if isinstance(extra_info, dict):
                extra_info = dict(extra_info)
                extra_info["num_turns"] = num_turns
                extra_info["rollout_reward_scores"] = rollout_reward_scores
                
            uid = item.non_tensor_batch["uid"]
            prepared.append({
                "i": i,
                "uid": uid,
                "data_source": data_source,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "extra_info": extra_info,
                "valid_response_length": int(valid_response_length),
                "valid_prompt_ids": valid_prompt_ids,  # 用于少量打印
            })
        
        # ----------------------------
        # 2) 并发 compute_score
        # ----------------------------
        sem = threading.Semaphore(self.max_inflight)

        stats = {
            "ok": 0,
            "fail": 0,
            "start_t": time.time(),
            "lat_ok": [],
            "lat_fail": [],
        }
        fail_printed = 0
        ok_printed = 0
        stats_lock = threading.Lock()

        # ✅ pairwise 需要：收集每条样本的 raw reward，然后按 uid 组内排序
        results = {}                 # i -> {"uid":..., "vlen":..., "raw":...}
        uid_groups = defaultdict(list)  # uid -> [i,...]
        
        def _run_one(payload: dict):
            """
            Return: (i, payload, score, err_str, latency)
            """
            t0 = time.time()
            try:
                with sem:
                    score = self.compute_score(
                        data_source=payload["data_source"],
                        solution_str=payload["response_str"],
                        ground_truth=payload["ground_truth"],
                        extra_info=payload["extra_info"],
                    )
                lat = time.time() - t0
                return payload["i"], payload, score, None, lat
            except Exception:
                lat = time.time() - t0
                err = traceback.format_exc()
                return payload["i"], payload, None, err, lat

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(_run_one, p) for p in prepared]

            for fut in as_completed(futures):
                i, payload, score, err, lat = fut.result()
                vlen = payload["valid_response_length"]
                uid = payload["uid"]

                if err is None:
                    # ✅ 成功
                    with stats_lock:
                        stats["ok"] += 1
                        stats["lat_ok"].append(lat)

                    if isinstance(score, dict):
                        raw_reward = float(score.get("score", 0.0))

                        # ✅ extra 里不要重复塞 "score"，避免一个 key 既当 reward 又当 metric
                        for k, v in score.items():
                            if k == "score":
                                continue
                            fv = _as_float(v)
                            if fv is not None:
                                reward_extra_info[k].append(fv)
                    else:
                        raw_reward = float(score)

                    # ✅ 这里不立刻写 reward_tensor（pairwise 要统一写）
                    results[i] = {"uid": uid, "vlen": vlen, "raw": raw_reward}
                    uid_groups[uid].append(i)

                    # ✅ 打印少量成功样例：确认并发在跑、耗时分布
                    if ok_printed < self.log_ok_samples:
                        ok_printed += 1
                        
                        if isinstance(score, dict):
                            try:
                                score_str = json.dumps(score, ensure_ascii=False)
                            except Exception:
                                score_str = str(score)
                        else:
                            score_str = str(score)
                            
                        print(f"[reward-ok] i={i} uid={uid} data_source={payload['data_source']} "
                              f"latency={lat:.3f}s raw_reward={raw_reward:.4f} score={score_str}")

                        # ✅ 只打印 1 个 pred/gt（每次 __call__ 仅一次）
                        if ok_printed == 1:
                            pred_preview = (payload["response_str"] or "").replace("\n", "\\n")
                            gt_preview = (payload["ground_truth"] or "").replace("\n", "\\n")
                            print(f"[reward-ok] pred_preview={pred_preview}")
                            print(f"[reward-ok] gt_preview={gt_preview}")

                else:
                    # ❌ 失败：打印信息 + raw_reward=0
                    with stats_lock:
                        stats["fail"] += 1
                        stats["lat_fail"].append(lat)

                    # ✅ 收集失败样本也进入组（避免组大小变少导致后续报错）
                    results[i] = {"uid": uid, "vlen": vlen, "raw": 0.0}
                    uid_groups[uid].append(i)

                    if fail_printed < self.log_fail_limit:
                        fail_printed += 1
                        gt_preview = (payload["ground_truth"] or "")[:200].replace("\n", "\\n")
                        pred_preview = (payload["response_str"] or "")[:200].replace("\n", "\\n")
                        print(f"[reward-fail] i={i} uid={uid} data_source={payload['data_source']} "
                              f"latency={lat:.3f}s -> raw_reward=0")
                        print(f"[reward-fail] pred_preview={pred_preview}")
                        print(f"[reward-fail] gt_preview={gt_preview}")
                        print("[reward-fail] traceback:\n", err)

                # ----------------------------
                # 3) 原有打印：每个 data_source 打印 num_examine 个
                # ----------------------------
                ds = payload["data_source"]
                already_print_data_sources.setdefault(ds, 0)
                if already_print_data_sources[ds] < self.num_examine:
                    already_print_data_sources[ds] += 1
                    prompt_str = self.tokenizer.decode(payload["valid_prompt_ids"], skip_special_tokens=True)
                    print("[prompt]", prompt_str)
                    print("[response]", payload["response_str"])
                    print("[ground_truth]", payload["ground_truth"])
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)

        # ----------------------------
        # 3.5) Pairwise: 按 uid 组内排序 -> 写回 reward_tensor
        # ----------------------------
        enable_pairwise = self.enable_pairwise and _is_train_phase(data)
        enable_pairwise = False

        if enable_pairwise:
            # ✅ 用“按 index 写入”的方式，保证所有 pairwise 指标都是长度=n（per-sample）
            pw_group_size = [0.0] * n
            pw_margin = [0.0] * n
            pw_borda_spread = [0.0] * n
            pw_raw_mean = [0.0] * n
            pw_raw_var = [0.0] * n

            # ✅ group_size_hist 也做成 per-sample：每个样本拿到自己所在组大小对应的 count（或者占比）
            # 这里用 count（与你原来的 v 一致），并广播到所有样本（每个样本都能看到同一个 hist 值）
            try:
                gs_hist = Counter([len(v) for v in uid_groups.values()])
                print("[pairwise] group_size_hist =", gs_hist)

                # 初始化每个 hist key 的 per-sample list
                gs_hist_ps = {k: [0.0] * n for k in gs_hist.keys()}
                for k, v in gs_hist.items():
                    # 广播：所有样本都记录本次 call 的 hist 计数
                    for ii in range(n):
                        gs_hist_ps[k][ii] = float(v)

                # 回填到 reward_extra_info（每个 key 都是长度=n）
                for k, arr in gs_hist_ps.items():
                    reward_extra_info[f"pairwise/group_size_hist/{k}"] = arr
            except Exception:
                pass

            printed_groups = 0  # ✅ 新增：控制每次 call 打印多少组
            for uid, idxs in uid_groups.items():
                if not idxs:
                    continue

                score_map = {ii: results[ii]["raw"] for ii in idxs}
                order, borda = _borda_order(idxs, score_map, tie_eps=self.pairwise_tie_eps)

                n_cand = len(order)

                # rank -> reward 写回
                for r, ii in enumerate(order):
                    vlen = results[ii]["vlen"]
                    if vlen > 0:
                        reward_tensor[ii, vlen - 1] = _rank_to_reward(r, n_cand)

                # ✅ 新增：打印 rank->reward
                if self.pairwise_print_groups > 0 and printed_groups < self.pairwise_print_groups:
                    printed_groups += 1
                    topk = min(self.pairwise_print_topk, n_cand)
                    print(f"[pairwise-rank] uid={uid} n_cand={n_cand} (show top {topk})")
                    for r, ii in enumerate(order[:topk]):
                        rr = _rank_to_reward(r, n_cand)
                        print(f"  r={r:02d} i={ii:03d} raw={score_map[ii]:.4f} rank_reward={rr:.4f}")
                    if n_cand > topk:
                        worst_i = order[-1]
                        print(f"  ... worst i={worst_i:03d} raw={score_map[worst_i]:.4f} rank_reward={_rank_to_reward(n_cand-1, n_cand):.4f}")

                # 计算 group-level 指标
                margin = 0.0
                if n_cand >= 2:
                    top1, top2 = order[0], order[1]
                    margin = float(score_map[top1] - score_map[top2])

                bvals = [borda[ii] for ii in idxs]
                spread = float(max(bvals) - min(bvals)) if bvals else 0.0

                raws = [score_map[ii] for ii in idxs]
                mean_raw = float(sum(raws) / max(len(raws), 1))
                var_raw = float(sum((x - mean_raw) ** 2 for x in raws) / max(len(raws), 1))

                gsz = float(len(idxs))

                # ✅ 广播到组内每个样本（按 index 写入，保证长度=n）
                for ii in idxs:
                    pw_group_size[ii] = gsz
                    pw_margin[ii] = margin
                    pw_borda_spread[ii] = spread
                    pw_raw_mean[ii] = mean_raw
                    pw_raw_var[ii] = var_raw

            # ✅ 最后统一塞回（长度全是 n）
            reward_extra_info["pairwise/group_size"] = pw_group_size
            reward_extra_info["pairwise/top12_margin"] = pw_margin
            reward_extra_info["pairwise/borda_spread"] = pw_borda_spread
            reward_extra_info["pairwise/raw_mean"] = pw_raw_mean
            reward_extra_info["pairwise/raw_var"] = pw_raw_var

        else:
            # 不启用 pairwise：保持原逻辑，用 raw_reward 直接写回
            for i, rec in results.items():
                vlen = rec["vlen"]
                if vlen > 0:
                    reward_tensor[i, vlen - 1] = float(rec["raw"])


        # ----------------------------
        # 4) 打印汇总：确认整体并发是否成功
        # ----------------------------
        if self.print_every_call_summary:
            elapsed = time.time() - stats["start_t"]
            ok = stats["ok"]
            fail = stats["fail"]
            tot = ok + fail
            avg_ok = (sum(stats["lat_ok"]) / len(stats["lat_ok"])) if stats["lat_ok"] else 0.0
            avg_fail = (sum(stats["lat_fail"]) / len(stats["lat_fail"])) if stats["lat_fail"] else 0.0
            print(f"[reward-summary] total={tot} ok={ok} fail={fail} "
                  f"elapsed={elapsed:.2f}s "
                  f"avg_ok={avg_ok:.3f}s avg_fail={avg_fail:.3f}s "
                  f"workers={self.max_workers} inflight={self.max_inflight} "
                  f"pairwise={enable_pairwise}")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
