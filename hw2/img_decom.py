import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from svd_decom import svd_from_scratch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# =========================================================
# 你已经实现好的函数：
# svd_from_scratch(A, tol=1e-10)
# 返回: U, Sigma, Vt, singular_values
# =========================================================


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
# 单通道 SVD 压缩
# =========================
def reconstruct_channel_with_k(U, Sigma, Vt, k):
    """
    利用前k个奇异值重建单通道图像
    """
    Uk = U[:, :k]
    Sk = Sigma[:k, :k]
    Vtk = Vt[:k, :]
    return Uk @ Sk @ Vtk


def compress_single_channel(channel, k, tol=1e-10):
    """
    对单通道矩阵进行 SVD 压缩
    返回:
        compressed_channel
        svd_info: 字典，包含分解结果和信息
    """
    U, Sigma, Vt, singular_values = svd_from_scratch(channel, tol=tol, k=k)

    max_rank = min(channel.shape[0], channel.shape[1])
    k = min(k, max_rank)

    compressed_channel = reconstruct_channel_with_k(U, Sigma, Vt, k)

    svd_info = {
        "U": U,
        "Sigma": Sigma,
        "Vt": Vt,
        "singular_values": singular_values,
        "k": k,
        "shape": channel.shape
    }
    return compressed_channel, svd_info


# =========================
# 彩色图像 SVD 压缩
# =========================
def compress_color_image(image_array, k, tol=1e-10):
    """
    对彩色图像的 R/G/B 三个通道分别压缩
    输入:
        image_array: shape=(H,W,3), float64
        k: 保留奇异值个数
    返回:
        compressed_image: 压缩后的彩色图像
        compression_info: 各通道SVD信息
    """
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("输入必须是 RGB 彩色图像，shape=(H,W,3)")

    compressed_channels = []
    channel_infos = []

    for c in range(3):
        channel = image_array[:, :, c]
        compressed_channel, info = compress_single_channel(channel, k, tol=tol)
        compressed_channels.append(compressed_channel)
        channel_infos.append(info)

    compressed_image = np.stack(compressed_channels, axis=2)
    compressed_image = clip_image(compressed_image)

    compression_info = {
        "R": channel_infos[0],
        "G": channel_infos[1],
        "B": channel_infos[2],
        "k": k,
        "original_shape": image_array.shape
    }

    return compressed_image, compression_info


# =========================
# 评价指标
# =========================
def calculate_mse(original, compressed):
    """
    MSE: 均方误差
    支持灰度或彩色
    """
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    return np.mean((original - compressed) ** 2)


def calculate_psnr(original, compressed, max_pixel=255.0):
    """
    PSNR
    """
    mse = calculate_mse(original, compressed)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_pixel ** 2) / mse)


def calculate_ssim_single_channel(img1, img2, max_pixel=255.0):
    """
    单通道全局 SSIM
    这是简化版全局SSIM，不是滑窗版，但足够适合课程实验/GUI演示
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
    彩色图像 SSIM:
    分别计算RGB三个通道的SSIM，再取平均
    """
    if original.ndim == 2:
        return calculate_ssim_single_channel(original, compressed, max_pixel)

    ssim_values = []
    for c in range(3):
        ssim_c = calculate_ssim_single_channel(
            original[:, :, c], compressed[:, :, c], max_pixel=max_pixel
        )
        ssim_values.append(ssim_c)

    return float(np.mean(ssim_values))


def calculate_compression_ratio(image_shape, k):
    """
    估算 SVD 压缩比（针对彩色图像）
    原图存储量: 3*m*n
    压缩后每个通道存储量: k*(m+n+1)
    三个通道总存储量: 3*k*(m+n+1)

    压缩比定义:
        原始存储量 / 压缩后存储量
    """
    h, w, ch = image_shape
    if ch != 3:
        raise ValueError("这里只处理 RGB 图像")

    original_size = 3 * h * w
    compressed_size = 3 * k * (h + w + 1)

    if compressed_size == 0:
        return float("inf")

    return original_size / compressed_size


def calculate_data_retention_ratio(image_shape, k):
    """
    数据保留率 = 压缩后存储量 / 原始存储量
    """
    ratio = calculate_compression_ratio(image_shape, k)
    if ratio == 0:
        return float("inf")
    return 1.0 / ratio


def calculate_energy_ratio(singular_values, k):
    """
    奇异值能量保留率:
        sum_{i=1}^k sigma_i^2 / sum sigma_i^2
    """
    singular_values = np.array(singular_values, dtype=np.float64)
    if singular_values.size == 0:
        return 0.0

    total_energy = np.sum(singular_values ** 2)
    if total_energy == 0:
        return 0.0

    k = min(k, len(singular_values))
    retained_energy = np.sum(singular_values[:k] ** 2)
    return retained_energy / total_energy


def evaluate_compression(original, compressed, compression_info):
    """
    统一计算多种评价指标
    返回字典，方便后续 GUI 直接调用
    """
    k = compression_info["k"]
    image_shape = compression_info["original_shape"]

    mse = calculate_mse(original, compressed)
    psnr = calculate_psnr(original, compressed)
    ssim = calculate_ssim_color(original, compressed)

    compression_ratio = calculate_compression_ratio(image_shape, k)
    data_retention_ratio = calculate_data_retention_ratio(image_shape, k)

    r_energy = calculate_energy_ratio(compression_info["R"]["singular_values"], k)
    g_energy = calculate_energy_ratio(compression_info["G"]["singular_values"], k)
    b_energy = calculate_energy_ratio(compression_info["B"]["singular_values"], k)
    avg_energy = (r_energy + g_energy + b_energy) / 3.0

    metrics = {
        "k": k,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "compression_ratio": compression_ratio,
        "data_retention_ratio": data_retention_ratio,
        "energy_ratio_R": r_energy,
        "energy_ratio_G": g_energy,
        "energy_ratio_B": b_energy,
        "energy_ratio_avg": avg_energy
    }

    return metrics


# =========================
# 可视化
# =========================
def show_original_and_compressed(original, compressed, title_suffix=""):
    """
    并排显示原图和压缩图
    """
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
    """
    打印评价指标
    """
    print("========== Compression Metrics ==========")
    print(f"k                      : {metrics['k']}")
    print(f"MSE                    : {metrics['mse']:.6f}")
    print(f"PSNR                   : {metrics['psnr']:.6f} dB")
    print(f"SSIM                   : {metrics['ssim']:.6f}")
    print(f"Compression Ratio      : {metrics['compression_ratio']:.6f}")
    print(f"Data Retention Ratio   : {metrics['data_retention_ratio']:.6f}")
    print(f"Energy Ratio (R)       : {metrics['energy_ratio_R']:.6f}")
    print(f"Energy Ratio (G)       : {metrics['energy_ratio_G']:.6f}")
    print(f"Energy Ratio (B)       : {metrics['energy_ratio_B']:.6f}")
    print(f"Average Energy Ratio   : {metrics['energy_ratio_avg']:.6f}")
    print("=========================================")


# =========================
# 面向GUI的统一接口
# =========================
def compress_and_evaluate_image(image_path, k, tol=1e-10, save_path=None, show=False):
    """
    给 GUI/上层逻辑调用的统一接口

    输入:
        image_path: 图像路径
        k: 保留奇异值个数
        tol: SVD容差
        save_path: 若不为None，则保存压缩图
        show: 是否显示图像

    返回:
        result: dict
            {
                "original_image": ...,
                "compressed_image": ...,
                "compression_info": ...,
                "metrics": ...
            }
    """
    original_uint8, original_float = load_color_image(image_path)

    compressed_image, compression_info = compress_color_image(
        original_float, k=k, tol=tol
    )

    metrics = evaluate_compression(original_float, compressed_image, compression_info)

    if save_path is not None:
        save_color_image(compressed_image, save_path)

    if show:
        show_original_and_compressed(original_uint8, compressed_image, title_suffix=f"(k={k})")

    result = {
        "original_image": original_uint8,
        "compressed_image": compressed_image.astype(np.uint8),
        "compression_info": compression_info,
        "metrics": metrics
    }

    return result


# =========================
# 批量实验：测试多个 k
# =========================
def batch_experiment(image_path, k_list, tol=1e-10):
    """
    对多个 k 做批量实验
    返回:
        results: list[dict]
    """
    results = []
    for k in k_list:
        result = compress_and_evaluate_image(
            image_path=image_path,
            k=k,
            tol=tol,
            save_path=None,
            show=False
        )
        results.append(result)
    return results


def print_batch_metrics(results):
    """
    打印批量实验结果
    """
    print(f"{'k':>6} {'MSE':>12} {'PSNR(dB)':>12} {'SSIM':>12} {'CR':>12} {'Energy':>12}")
    for result in results:
        m = result["metrics"]
        print(f"{m['k']:>6} "
              f"{m['mse']:>12.4f} "
              f"{m['psnr']:>12.4f} "
              f"{m['ssim']:>12.6f} "
              f"{m['compression_ratio']:>12.4f} "
              f"{m['energy_ratio_avg']:>12.6f}")


# =========================
# 示例主函数
# =========================
def main():
    image_path = "G:\\USTC_MM_26\\hw2\\1.jpg"          # 改成你的图片路径
    save_path = "G:\\USTC_MM_26\\hw2\\1_compressed.jpg" # 保存路径
    k = 50

    result = compress_and_evaluate_image(
        image_path=image_path,
        k=k,
        tol=1e-10,
        save_path=save_path,
        show=True
    )

    print_metrics(result["metrics"])

    # 批量实验
    k_list = [5, 10, 20, 40, 80, 120]
    batch_results = batch_experiment(image_path, k_list)
    print_batch_metrics(batch_results)


if __name__ == "__main__":
    main()