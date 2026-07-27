import os
import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim_metric
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
import lpips
from tqdm import tqdm

class ImageMetrics:
    def __init__(self, device="cuda"):
        """
        初始化指标类，加载 LPIPS 和 Inception 模型
        """
        self.device = device
        
        # 1. 加载 LPIPS
        self.lpips_model = lpips.LPIPS(net="vgg").to(device)
        self.lpips_model.eval()

        # 2. 加载 InceptionV3 用于特征提取
        # 我们替换最后的全连接层(fc)为 Identity，以便直接获取 2048 维特征
        weights = Inception_V3_Weights.DEFAULT
        self.inception_model = inception_v3(weights=weights, transform_input=False).to(device)
        self.inception_model.fc = nn.Identity() 
        self.inception_model.eval()

        # Inception 预处理标准
        self.inception_preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # ---------- 提取 Inception 特征 ----------
    def get_inception_feature(self, img):
        """
        输入 PIL Image，返回 (2048,) 的 numpy 向量
        """
        img_tensor = self.inception_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Inception v3 forward 返回 logits，我们需要处理一下
            # 当 model.fc 被替换为 Identity 后，输出即为 pooling 后的特征
            feature = self.inception_model(img_tensor)
        return feature.cpu().numpy().flatten()

    # ---------- 基础指标计算 (单对图片) ----------
    def mse(self, arr1, arr2):
        return np.mean((arr1 - arr2) ** 2)

    def psnr(self, arr1, arr2):
        m = self.mse(arr1, arr2)
        if m == 0: return float("inf")
        return 20 * math.log10(255.0 / math.sqrt(m))

    def ssim(self, arr1, arr2):
        return ssim_metric(arr1, arr2, channel_axis=2, data_range=255.0)

    def calc_single_pair_metrics(self, img1, img2):
        """
        计算一对图片的基础指标 (PSNR, SSIM, LPIPS)
        """
        # 尺寸对齐
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)
        
        arr1 = np.array(img1).astype("float32")
        arr2 = np.array(img2).astype("float32")

        # LPIPS 需要 Tensor [0,1]
        t1 = torch.tensor(arr1 / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        t2 = torch.tensor(arr2 / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        lpips_val = self.lpips_model(t1, t2).item()

        return {
            "psnr": self.psnr(arr1, arr2),
            "ssim": self.ssim(arr1, arr2),
            "lpips": lpips_val
        }

    # ---------- FID 计算 (基于特征矩阵) ----------
    def calc_fid_from_features(self, feats_real, feats_gen):
        """
        输入两个 (N, 2048) 的 numpy 矩阵，计算 FID
        """
        # 至少需要 2 个样本才能计算协方差
        if len(feats_real) < 2 or len(feats_gen) < 2:
            print("Warning: 样本数少于2，无法计算 FID，返回 -1")
            return -1.0

        mu1 = np.mean(feats_real, axis=0)
        sigma1 = np.cov(feats_real, rowvar=False)

        mu2 = np.mean(feats_gen, axis=0)
        sigma2 = np.cov(feats_gen, rowvar=False)

        ssdiff = np.sum((mu1 - mu2)**2.0)
        
        # 计算协方差矩阵乘积的平方根
        covmean = sqrtm(sigma1.dot(sigma2))

        # 数值稳定性处理：如果结果包含复数，取实部
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return float(fid)


def evaluate_qwen_outputs(root_dir, metrics_engine, methods):
    method_filenames = [f"{method}.png" for method in methods]
    
    # 存储单图指标结果
    results_per_metric = {method: {"psnr": [], "ssim": [], "lpips": []} for method in methods}
    
    # 存储特征向量用于最后计算 FID
    # 结构: {"next": [feat1, feat2...], "method_A": [feat1, feat2...]}
    # 注意：我们也需要收集 Ground Truth 的特征
    all_features = {method: [] for method in methods}
    gt_features = [] 
    
    valid_dirs = []
    
    # 1. 扫描所有有效目录
    for dirpath, _, filenames in os.walk(root_dir):
        # 必须包含 GT (next.png) 且至少包含一种待评测方法的文件
        if "next.png" not in filenames:
            continue
            
        # 检查是否包含我们要评测的方法图片
        # has_any_method = any(f in filenames for f in method_filenames)
        has_all_method = all(os.path.exists(os.path.join(dirpath, f)) for f in method_filenames)
        if not has_all_method:
            continue
            
        valid_dirs.append(dirpath)

    print(f"Found {len(valid_dirs)} valid directories. Starting evaluation...")

    # 2. 遍历处理
    for dirpath in tqdm(valid_dirs):
        gt_path = os.path.join(dirpath, "next.png")
        try:
            gt_img = Image.open(gt_path).convert("RGB")
        except Exception as e:
            print(f"Error reading GT {gt_path}: {e}")
            continue

        # 提取 GT 特征
        feat_gt = metrics_engine.get_inception_feature(gt_img)
        gt_features.append(feat_gt)

        # 遍历每个方法
        for method, filename in zip(methods, method_filenames):
            method_path = os.path.join(dirpath, filename)
            
            if not os.path.exists(method_path):
                continue
                
            try:
                pred_img = Image.open(method_path).convert("RGB")
            except Exception as e:
                print(f"Error reading {method_path}: {e}")
                continue

            # A. 计算单图指标 (PSNR, SSIM, LPIPS)
            scores = metrics_engine.calc_single_pair_metrics(gt_img, pred_img)
            for k, v in scores.items():
                results_per_metric[method][k].append(v)
            
            # B. 提取特征，存入列表
            feat_pred = metrics_engine.get_inception_feature(pred_img)
            all_features[method].append(feat_pred)

    # 3. 汇总计算
    final_report = {}

    # 将 GT 特征转为 numpy 矩阵 (N, 2048)
    np_gt_feats = np.array(gt_features)

    for method in methods:
        final_report[method] = {}
        
        # 平均 PSNR/SSIM/LPIPS
        for k in ["psnr", "ssim", "lpips"]:
            vals = results_per_metric[method][k]
            if vals:
                final_report[method][k] = sum(vals) / len(vals)
            else:
                final_report[method][k] = 0.0
        
        # 计算 FID (分布级指标)
        np_pred_feats = np.array(all_features[method])
        
        if len(np_gt_feats) > 1 and len(np_pred_feats) > 1:
            print(f"Calculating FID for {method} (samples: {len(np_pred_feats)})...")
            fid_score = metrics_engine.calc_fid_from_features(np_gt_feats, np_pred_feats)
            final_report[method]["fid"] = fid_score
        else:
            final_report[method]["fid"] = -1.0  # 样本不足

    return final_report


if __name__ == "__main__":
    # 初始化
    metrics_engine = ImageMetrics(device="cuda")
    
    root_directory = os.environ.get("DATA_ROOT", "data") + "/new_prompt/test_mini"
    
    # 待评测的方法列表 (文件名不含后缀)
    # target_methods = [
    #     "next_base_prev_action_prompt1226",
    #     "next_base_sftvl_prev_action_prompt1226",
    #     "next_base_qwenvl_pred_prompt",
    #     "next_base_pred_prompt1226", 
    #     "next_qwenvlsft_prev_action_prompt1226_epoch_2_qwen_prompt_prev_action_prompt1226",
    #     "next_qwenvlsft_pred_prompt1226_epoch_2_qwen_prompt_qwenvl_lora_pred_prompt1226",
    #     "next_qwenvlsft_pred_prompt1226_freezevl_infer_epoch_2_qwen_prompt_qwenvl_lora_pred_prompt1226",
    #     "next_gpt_epoch_10_qwen_prompt"
    # ]
    target_methods = []
    # for model in ["base", "epoch-24"]:
    #     for prompt in ["action_only", "action_with_description", "nl_gt", "nl_pred_checkpoint-450-merged", "nl_pred_gpt-5.2-chat-20251211", "nl_pred_base"]:
    #         method_name = f"generated_next/next_{model}_prompt_{prompt}"
    #         target_methods.append(method_name)
    
    target_methods = [f"generated_next/next_gpt-image-1.5_prompt_nl_gt"]
    
    # 如果你也想评测 "next" (即 GT 本身，应该全是满分/0距离)，可以加进去
    # target_methods.append("next") 

    results = evaluate_qwen_outputs(root_directory, metrics_engine, methods=target_methods)
    
    print("\n========== 评估结果 Evaluation Summary ==========\n")
    
    # 打印表头
    metrics_keys = ["psnr", "ssim", "lpips", "fid"]
    header = ["Method"] + metrics_keys
    
    # 打印格式化表格
    row_format = "{:<50}" + "{:>15}" * len(metrics_keys)
    print(row_format.format(*header))
    print("-" * (50 + 15 * len(metrics_keys)))
    
    for method in target_methods:
        if method not in results: continue
        row_data = [method]
        for mk in metrics_keys:
            val = results[method].get(mk, -1)
            row_data.append(f"{val:.4f}")
        print(row_format.format(*row_data))
    
    print("\n=================================================\n")