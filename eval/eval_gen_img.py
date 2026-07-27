import os
import math

import numpy as np
import torch
from PIL import Image
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim_metric
from torchvision import transforms
from torchvision.models import inception_v3
import lpips
from tqdm import tqdm
import json

class ImageMetrics:
    def __init__(self, device="cuda"):
        """
        初始化指标类，自动加载 LPIPS 模型
        """
        self.device = device
        self.lpips_model = lpips.LPIPS(net="vgg").to(device)

    # ---------- 工具函数 ----------
    def _load_pair(self, img1_path, img2_path):
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if img1.size != img2.size:
            img2 = img2.resize(img1.size)

        return img1, img2

    # ---------- 差异比例 ----------
    def diff_ratio(self, img1, img2):
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        diff = (arr1 != arr2).sum()
        total = arr1.size
        ratio = diff / total
        return float(ratio)

    # ---------- MSE ----------
    def mse(self, img1, img2):
        arr1 = np.array(img1).astype("float32")
        arr2 = np.array(img2).astype("float32")
        return float(np.mean((arr1 - arr2) ** 2))

    # ---------- PSNR ----------
    def psnr(self, img1, img2):
        m = self.mse(img1, img2)
        if m == 0:
            return float("inf")
        MAX = 255.0
        return 20 * math.log10(MAX / math.sqrt(m))

    # ---------- SSIM ----------
    def ssim(self, img1, img2):
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        ssim_val, _ = ssim_metric(arr1, arr2, channel_axis=2, full=True)
        return float(ssim_val)

    # ---------- LPIPS ----------
    def lpips(self, img1, img2):
        t1 = torch.tensor(np.array(img1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t2 = torch.tensor(np.array(img2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t1 = t1.to(self.device)
        t2 = t2.to(self.device)
        return self.lpips_model(t1, t2).item()

     # ---------- FID ----------
    def fid(self, img1, img2):
        """
        计算两张图片之间的FID (简化版本,使用InceptionV3特征)
        注意: 标准FID需要多张图片的分布,这里是单对图片的近似
        """
        
        # 加载预训练模型
        if not hasattr(self, 'inception_model'):
            self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
            self.inception_model.eval()
        
        # 预处理图片
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        t1 = preprocess(img1).unsqueeze(0).to(self.device)
        t2 = preprocess(img2).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            feat1 = self.inception_model(t1).cpu().numpy().flatten()
            feat2 = self.inception_model(t2).cpu().numpy().flatten()
        
        # 计算均值和协方差
        mu1, sigma1 = feat1.mean(), np.cov(feat1.reshape(1, -1), rowvar=False)
        mu2, sigma2 = feat2.mean(), np.cov(feat2.reshape(1, -1), rowvar=False)
        
        # FID公式
        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = sqrtm(sigma1 * sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid_score = ssdiff + np.trace(sigma1 + sigma2 - 2.0*covmean)
        return float(fid_score)

    # ---------- 一次性计算全部 ----------
    def calc_all_metrics(self, img1=None, img2=None, img1_path=None, img2_path=None):
        if img1 is None and img1_path is not None:
            img1 = Image.open(img1_path).convert("RGB")
        if img2 is None and img2_path is not None:
            img2 = Image.open(img2_path).convert("RGB")
        
        # 统一处理尺寸不一致的情况
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)
        
        return {
            "diff_ratio": self.diff_ratio(img1, img2),
            "mse": self.mse(img1, img2),
            "psnr": self.psnr(img1, img2),
            # "ssim": self.ssim(img1, img2),
            # "lpips": self.lpips(img1, img2),
            # "fid": self.fid(img1, img2),
        }
    
    def calc_scores(self, data):
        all_metrics = []
        
        for gt, pred in data:
            if isinstance(gt, str):
                gt = Image.open(gt).convert("RGB")
            if isinstance(pred, str):
                pred = Image.open(pred).convert("RGB")
            
            metrics = self.calc_all_metrics(img1=pred, img2=gt)
            all_metrics.append(metrics)
        
        if not all_metrics:
            return {}

        avg_metrics = {}
        metric_keys = all_metrics[0].keys()
        
        for key in metric_keys:
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        return avg_metrics
        
    
   

def evaluate_qwen_outputs(root_dir, metrics, methods=["next", "next_base", "next_lora_epoch_10"]):
    # 动态初始化结果字典
    method_filenames = [f"generated_next/{method}.png" for method in methods]
    results = {method: [] for method in methods}
    
    # 先找到所有valid的folder
    valid_folders = []
    print("扫描所有文件夹...")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "next.png" in filenames:
            valid_dir = True
            for filename in method_filenames:
                # print(os.path.join(dirpath, filename))
                if not os.path.exists(os.path.join(dirpath, filename)):
                    
                    valid_dir = False
                    break
            if valid_dir:
                valid_folders.append(dirpath)
    
    print(f"找到 {len(valid_folders)} 个有效文件夹")
    
    # 遍历所有valid的folder
    for dirpath in tqdm(valid_folders):
        gt_path = os.path.join(dirpath, "next.png")
        per_dir_paths = {}

        for method, filename in zip(methods, method_filenames):
            per_dir_paths[method] = os.path.join(dirpath, filename)

        # 计算每个 method 的得分
        for method, method_path in per_dir_paths.items():
            score = metrics.calc_all_metrics(img1_path=method_path, img2_path=gt_path)
            results[method].append(score)
    
    print(f"共处理 {len(valid_folders)} 个目录。")

    # ------ 统计平均 ------
    def average_results(result_list):
        if not result_list:
            return {}

        keys = result_list[0].keys()
        return {
            key: sum(item[key] for item in result_list) / len(result_list)
            for key in keys
        }

    # 平均化
    avg_results = {
        method: average_results(score_list)
        for method, score_list in results.items()
    }

    return avg_results



if __name__ == "__main__":
    metrics = ImageMetrics(device="cuda")
    save_path = "output/eval_gen_img_results_gpt.json"
    
    # 加载已有结果
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    root_directory = os.environ.get("DATA_ROOT", "data") + "/new_prompt/test_mini"
    
    methods = [f"next_gpt-image-1.5_prompt_nl_gt"]
    # for model in ["base", "epoch-24", "gpt-image-1.5"]:
    #     for prompt in ["action_only", "action_with_description", "nl_gt", "nl_pred_checkpoint-450-merged", "nl_pred_gpt-5.2-chat-20251211", "nl_pred_base"]:
    #         method_name = f"next_{model}_prompt_{prompt}"
    #         methods.append(method_name)
    
    
    
    # 找出需要重新跑的方法
    # methods_to_run = [m for m in methods if m not in results]
    methods_to_run = methods
    
    if methods_to_run:
        print(f"需要评估的方法: {methods_to_run}")
        new_results = evaluate_qwen_outputs(root_directory, metrics, methods=methods_to_run)
        # 合并到已有结果中
        results.update(new_results)
        
        # 保存更新后的结果
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"结果已保存到: {save_path}")
    else:
        print("所有方法都已有结果，跳过评估")
    
    print("results:", results)

    print("\n========== 评估结果 Evaluation Summary ==========\n")

    # 取所有方法（列）与所有指标（行）
    methods = list(results.keys())
    metrics = list(next(iter(results.values())).keys())
    

    # 表头
    header = ["Metric"] + methods
    print("{:<15}".format(header[0]), end="")
    for m in methods:
        print("{:>15}".format(m), end="")
    print()

    # 表体
    for metric in metrics:
        print("{:<15}".format(metric), end="")
        for m in methods:
            value = results[m].get(metric, None)
            if value is None:
                text = "-"
            else:
                text = f"{value:.6f}"
            print("{:>15}".format(text), end="")
        print()

    print("\n=================================================\n")