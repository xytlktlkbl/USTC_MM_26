import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys


# =========================
# 图像读写与基础处理
# =========================

def load_color_image(image_path):
    """
    读取彩色图像，返回:
        img_uint8: uint8格式, shape=(H, W, 3)
        img_float: float64格式, shape=(H, W, 3)
    """
    img = Image.open(image_path).convert("RGB")
    img_uint8 = np.array(img, dtype=np.uint8)
    img_float = img_uint8.astype(np.float64)
    return img_uint8, img_float


def save_color_image(image_array, save_path):
    """
    保存彩色图像，输入应为 [0,255] 范围
    """
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    Image.fromarray(image_array).save(save_path)


def clip_image(image_array):
    """
    将图像裁剪到 [0,255]
    """
    return np.clip(image_array, 0, 255)


# =========================
# 小波系数阈值处理
# =========================

def apply_threshold_to_coeffs(coeffs, threshold):
    """
    对多层小波系数列表应用阈值：绝对值小于 threshold 的系数置零
    coeffs 结构：[cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
    """
    result = [coeffs[0]]  # 保留最低频近似系数 cA_n（不阈值化）
    for detail_tuple in coeffs[1:]:
        thresholded = tuple(
            np.where(np.abs(band) < threshold, 0.0, band)
            for band in detail_tuple
        )
        result.append(thresholded)
    return result


def count_nonzero_coeffs(coeffs):
    """
    统计小波系数中非零元素的总数（用于估算稀疏度）
    """
    total = 0
    nonzero = 0
    # 近似子带
    total += coeffs[0].size
    nonzero += np.count_nonzero(coeffs[0])
    # 细节子带
    for detail_tuple in coeffs[1:]:
        for band in detail_tuple:
            total += band.size
            nonzero += np.count_nonzero(band)
    return nonzero, total


def calculate_energy_retention(original_coeffs, thresholded_coeffs):
    """
    计算阈值后小波系数能量保留率（仅统计细节子带）
    energy_ratio = sum(thresholded^2) / sum(original^2)
    """
    orig_energy = 0.0
    ret_energy = 0.0

    # 全部子带（包含近似子带）
    all_orig = [original_coeffs[0]] + [b for dt in original_coeffs[1:] for b in dt]
    all_thresh = [thresholded_coeffs[0]] + [b for dt in thresholded_coeffs[1:] for b in dt]

    for o, t in zip(all_orig, all_thresh):
        orig_energy += np.sum(o ** 2)
        ret_energy += np.sum(t ** 2)

    if orig_energy == 0:
        return 1.0
    return ret_energy / orig_energy


# =========================
# 单通道 DWT 压缩
# =========================

def compress_single_channel(channel, threshold, wavelet="haar", level=1):
    """
    对单通道矩阵进行 DWT 压缩
    返回:
        compressed_channel: 重建后的通道（float64）
        dwt_info: 包含系数和相关信息的字典
    """
    # 小波分解
    coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=level)

    # 阈值化
    coeffs_thresholded = apply_threshold_to_coeffs(coeffs, threshold)

    # 计算能量保留率
    energy_ratio = calculate_energy_retention(coeffs, coeffs_thresholded)

    # 统计非零系数
    nonzero, total = count_nonzero_coeffs(coeffs_thresholded)

    # 重建
    compressed_channel = pywt.waverec2(coeffs_thresholded, wavelet=wavelet)

    # 裁剪到与原始通道相同尺寸（小波重建可能有 1 像素误差）
    h, w = channel.shape
    compressed_channel = compressed_channel[:h, :w]

    dwt_info = {
        "coeffs_original": coeffs,
        "coeffs_thresholded": coeffs_thresholded,
        "energy_ratio": energy_ratio,
        "nonzero_coeffs": nonzero,
        "total_coeffs": total,
        "sparsity": 1.0 - nonzero / total if total > 0 else 0.0,
        "shape": channel.shape,
        "wavelet": wavelet,
        "level": level,
        "threshold": threshold,
    }
    return compressed_channel, dwt_info


# =========================
# 彩色图像 DWT 压缩
# =========================

def compress_color_image(image_array, threshold, wavelet="haar", level=1):
    """
    对彩色图像的 R/G/B 三个通道分别进行 DWT 压缩
    输入:
        image_array: shape=(H, W, 3), float64
        threshold:   小波系数阈值
        wavelet:     小波函数名称
        level:       分解层数
    返回:
        compressed_image: 压缩后的彩色图像 (float64, clipped)
        compression_info: 各通道 DWT 信息及全局参数
    """
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("输入必须是 RGB 彩色图像，shape=(H,W,3)")

    compressed_channels = []
    channel_infos = []
    channel_names = ["R", "G", "B"]

    for c in range(3):
        channel = image_array[:, :, c]
        compressed_channel, info = compress_single_channel(
            channel, threshold=threshold, wavelet=wavelet, level=level
        )
        compressed_channels.append(compressed_channel)
        channel_infos.append(info)

    compressed_image = np.stack(compressed_channels, axis=2)
    compressed_image = clip_image(compressed_image)

    compression_info = {
        "R": channel_infos[0],
        "G": channel_infos[1],
        "B": channel_infos[2],
        "threshold": threshold,
        "wavelet": wavelet,
        "level": level,
        "original_shape": image_array.shape,
    }

    return compressed_image, compression_info


# =========================
# 评价指标
# =========================

def calculate_mse(original, compressed):
    """MSE：均方误差，支持灰度或彩色"""
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    return np.mean((original - compressed) ** 2)


def calculate_psnr(original, compressed, max_pixel=255.0):
    """PSNR（峰值信噪比，单位 dB）"""
    mse = calculate_mse(original, compressed)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_pixel ** 2) / mse)


def calculate_ssim_single_channel(img1, img2, max_pixel=255.0):
    """
    单通道全局 SSIM（简化版，不使用滑窗）
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    sigma1_sq = np.mean((img1 - mu1) ** 2)
    sigma2_sq = np.mean((img2 - mu2) ** 2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    C1 = (0.01 * max_pixel) ** 2
    C2 = (0.03 * max_pixel) ** 2

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    if denominator == 0:
        return 1.0
    return numerator / denominator


def calculate_ssim_color(original, compressed, max_pixel=255.0):
    """
    彩色图像 SSIM：分别计算 RGB 三通道再取平均
    """
    if original.ndim == 2:
        return calculate_ssim_single_channel(original, compressed, max_pixel)

    ssim_values = [
        calculate_ssim_single_channel(
            original[:, :, c], compressed[:, :, c], max_pixel=max_pixel
        )
        for c in range(3)
    ]
    return float(np.mean(ssim_values))


def calculate_compression_ratio(compression_info):
    """
    估算 DWT 压缩比（基于非零系数稀疏度）

    压缩比定义：
        原始像素总数 / 压缩后非零系数总数（三通道之和）
    """
    h, w, _ = compression_info["original_shape"]
    original_size = 3 * h * w

    total_nonzero = sum(
        compression_info[ch]["nonzero_coeffs"]
        for ch in ["R", "G", "B"]
    )

    if total_nonzero == 0:
        return float("inf")

    return original_size / total_nonzero


def calculate_data_retention_ratio(compression_info):
    """
    数据保留率 = 1 / 压缩比
    """
    ratio = calculate_compression_ratio(compression_info)
    if ratio == 0 or ratio == float("inf"):
        return 0.0
    return 1.0 / ratio


def calculate_sparsity(compression_info):
    """
    稀疏度：三通道平均非零率
    """
    sparsities = [
        compression_info[ch]["sparsity"]
        for ch in ["R", "G", "B"]
    ]
    return float(np.mean(sparsities))


def evaluate_compression(original, compressed, compression_info):
    """
    统一计算多种评价指标，返回字典（方便 GUI 直接调用）
    """
    mse = calculate_mse(original, compressed)
    psnr = calculate_psnr(original, compressed)
    ssim = calculate_ssim_color(original, compressed)

    compression_ratio = calculate_compression_ratio(compression_info)
    data_retention_ratio = calculate_data_retention_ratio(compression_info)
    sparsity = calculate_sparsity(compression_info)

    r_energy = compression_info["R"]["energy_ratio"]
    g_energy = compression_info["G"]["energy_ratio"]
    b_energy = compression_info["B"]["energy_ratio"]
    avg_energy = (r_energy + g_energy + b_energy) / 3.0

    metrics = {
        "threshold": compression_info["threshold"],
        "wavelet": compression_info["wavelet"],
        "level": compression_info["level"],
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "compression_ratio": compression_ratio,
        "data_retention_ratio": data_retention_ratio,
        "sparsity": sparsity,
        "energy_ratio_R": r_energy,
        "energy_ratio_G": g_energy,
        "energy_ratio_B": b_energy,
        "energy_ratio_avg": avg_energy,
    }

    return metrics


# =========================
# 可视化
# =========================

def show_original_and_compressed(original, compressed, title_suffix=""):
    """并排显示原图和压缩图"""
    original = clip_image(original).astype(np.uint8)
    compressed = clip_image(compressed).astype(np.uint8)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(compressed)
    plt.title(f"Compressed Image {title_suffix}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def print_metrics(metrics):
    """打印评价指标"""
    print("========== DWT Compression Metrics ==========")
    print(f"Wavelet                : {metrics['wavelet']}")
    print(f"Level                  : {metrics['level']}")
    print(f"Threshold              : {metrics['threshold']}")
    print(f"MSE                    : {metrics['mse']:.6f}")
    print(f"PSNR                   : {metrics['psnr']:.6f} dB")
    print(f"SSIM                   : {metrics['ssim']:.6f}")
    print(f"Compression Ratio      : {metrics['compression_ratio']:.6f}")
    print(f"Data Retention Ratio   : {metrics['data_retention_ratio']:.6f}")
    print(f"Sparsity               : {metrics['sparsity']:.6f}")
    print(f"Energy Ratio (R)       : {metrics['energy_ratio_R']:.6f}")
    print(f"Energy Ratio (G)       : {metrics['energy_ratio_G']:.6f}")
    print(f"Energy Ratio (B)       : {metrics['energy_ratio_B']:.6f}")
    print(f"Average Energy Ratio   : {metrics['energy_ratio_avg']:.6f}")
    print("=============================================")


# =========================
# 面向 GUI 的统一接口
# =========================

def compress_and_evaluate_image(
    image_path,
    threshold=10.0,
    wavelet="haar",
    level=1,
    save_path=None,
    show=False,
):
    """
    供 GUI / 上层逻辑调用的统一接口

    参数:
        image_path : 图像路径
        threshold  : 小波系数阈值（绝对值小于此值的系数置零）
        wavelet    : 小波函数名称（如 "haar", "db2", "sym4", "bior2.2"）
        level      : 小波分解层数（建议 1~5）
        save_path  : 若不为 None，则保存压缩后的图像
        show       : 是否弹窗显示图像对比

    返回:
        result: dict
            {
                "original_image"   : np.ndarray uint8, shape=(H,W,3)
                "compressed_image" : np.ndarray uint8, shape=(H,W,3)
                "compression_info" : dict  （各通道系数、参数等）
                "metrics"          : dict  （MSE/PSNR/SSIM/CR/Energy/…）
            }
    """
    original_uint8, original_float = load_color_image(image_path)

    compressed_image, compression_info = compress_color_image(
        original_float,
        threshold=threshold,
        wavelet=wavelet,
        level=level,
    )

    metrics = evaluate_compression(original_float, compressed_image, compression_info)

    if save_path is not None:
        save_color_image(compressed_image, save_path)

    if show:
        show_original_and_compressed(
            original_uint8,
            compressed_image,
            title_suffix=f"(wavelet={wavelet}, level={level}, thr={threshold})",
        )

    result = {
        "original_image": original_uint8,
        "compressed_image": compressed_image.astype(np.uint8),
        "compression_info": compression_info,
        "metrics": metrics,
    }

    return result


# =========================
# 批量实验：测试多个 threshold
# =========================

def batch_experiment(image_path, threshold_list, wavelet="haar", level=1):
    """
    对多个 threshold 做批量实验
    返回:
        results: list[dict]
    """
    results = []
    for thr in threshold_list:
        result = compress_and_evaluate_image(
            image_path=image_path,
            threshold=thr,
            wavelet=wavelet,
            level=level,
            save_path=None,
            show=False,
        )
        results.append(result)
    return results


def print_batch_metrics(results):
    """打印批量实验结果"""
    print(
        f"{'Threshold':>12} {'MSE':>12} {'PSNR(dB)':>12} "
        f"{'SSIM':>10} {'CR':>10} {'Energy':>10} {'Sparsity':>10}"
    )
    for result in results:
        m = result["metrics"]
        print(
            f"{m['threshold']:>12.4f} "
            f"{m['mse']:>12.4f} "
            f"{m['psnr']:>12.4f} "
            f"{m['ssim']:>10.6f} "
            f"{m['compression_ratio']:>10.4f} "
            f"{m['energy_ratio_avg']:>10.6f} "
            f"{m['sparsity']:>10.6f}"
        )


# =========================
# 示例主函数
# =========================

def main():
    image_path = "G:\\USTC_MM_26\\hw2\\1.jpg"           # 改成你的图片路径
    save_path  = "G:\\USTC_MM_26\\hw2\\1_dwt_compressed.jpg"

    result = compress_and_evaluate_image(
        image_path=image_path,
        threshold=10.0,
        wavelet="haar",
        level=2,
        save_path=save_path,
        show=True,
    )

    print_metrics(result["metrics"])

    # 批量实验
    #threshold_list = [1, 5, 10, 20, 50, 100, 200]
    #batch_results = batch_experiment(image_path, threshold_list, wavelet="haar", level=2)
    #print_batch_metrics(batch_results)


if __name__ == "__main__":
    main()
