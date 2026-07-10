from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import cv2
import numpy as np

from .utils import Shift, convert_shifts_to_hr, to_float01

#高频真值->镜头散焦->马达平移->像素积分->量子与电路噪声
"""
第一步（视觉说服）： 贴出局部放大对比图（实拍单帧放大 vs IBP vs RCAN），并附上像论文图 3 那样的“像素亮度截面曲线”，让人一眼看出锐度的提升。

第二步（客观打分）： 因为没有 GT，所以你不标 PSNR，而是标出 LaplacianSharpness 和 重投影 MSE。

第三步（工业价值）： 拿这些图去做最终的 LED 边缘检测或缺陷识别，用下游检测的准确率来一锤定音。
"""
@dataclass
class DegradationConfig:
    scale: int = 2
    shifts: Sequence[Shift] | None = None
    shift_unit: str = "lr"

    # --- 新增/升级的物理模拟参数 ---
    blur_type: str = "defocus"  # 'defocus' (散焦模糊, 推荐) 或 'gaussian' (高斯模糊)
    blur_kernel: int = 5  # 模糊核大小 (如果是 defocus，kernel=5 代表半径为2的圆盘)
    blur_sigma: float = 0.0  # 仅对 gaussian 有效
    blur_stage: str = "before_downsample"  # 强制默认在下采样前模糊 (符合真实物理光学)

    noise_poisson_a: float = 0.001  # 泊松噪声系数 (光子散粒噪声，与信号强度相关)
    noise_gaussian_b: float = 0.0001  # 高斯噪声方差 (传感器读取底噪，与信号无关)
    # -------------------------------

    seed: int | None = 0
    border_mode: str = "replicate"
    interpolation: str = "area"


def _border_mode(name: str) -> int:
    name = name.lower()
    if name == "reflect": return cv2.BORDER_REFLECT_101
    if name == "constant": return cv2.BORDER_CONSTANT
    return cv2.BORDER_REPLICATE


def _interp(name: str) -> int:
    name = name.lower()
    return {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }.get(name, cv2.INTER_AREA)


def gaussian_blur(img: np.ndarray, kernel: int = 3, sigma: float = 0.0) -> np.ndarray:
    if kernel is None or kernel <= 1: return img
    if kernel % 2 == 0: kernel += 1
    return cv2.GaussianBlur(img, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)


def defocus_blur(img: np.ndarray, kernel_size: int = 5, sigma: float = 0.0) -> np.ndarray:
    """模拟真实镜头散焦的圆盘模糊 (Pillbox Blur)"""
    if kernel_size is None or kernel_size <= 1: return img
    if kernel_size % 2 == 0: kernel_size += 1

    radius = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    # 画一个实心圆盘作为卷积核
    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
    kernel /= kernel.sum()  # 能量归一化

    return cv2.filter2D(img, -1, kernel)


def add_poisson_gaussian_noise(img: np.ndarray, a: float, b: float, rng: np.random.Generator) -> np.ndarray:
    """添加符合传感器物理特性的 泊松-高斯 联合噪声"""
    if a <= 0 and b <= 0: return img

    # 计算当前像素的理论方差：Variance = a * Signal + b
    # np.maximum 防止出现极微小的负值导致 sqrt 报错
    variance = np.maximum(img, 0.0) * a + b

    # 用局部高斯近似生成泊松-高斯噪声 (这是极高效率的科研标准做法)
    noise = rng.normal(0.0, np.sqrt(variance), size=img.shape).astype(np.float32)

    return np.clip(img + noise, 0.0, 1.0)


def warp_hr(img: np.ndarray, shift_hr: Shift, border_mode: str = "replicate") -> np.ndarray:
    h, w = img.shape[:2]
    M = np.array([[1.0, 0.0, shift_hr.dx], [0.0, 1.0, shift_hr.dy]], dtype=np.float32)
    return cv2.warpAffine(
        img.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=_border_mode(border_mode)
    )


def downsample(img: np.ndarray, scale: int, interpolation: str = "area") -> np.ndarray:
    h, w = img.shape[:2]
    if h % scale != 0 or w % scale != 0:
        raise ValueError(f"HR size {(h, w)} must be divisible by scale={scale}.")
    return cv2.resize(img, (w // scale, h // scale), interpolation=_interp(interpolation))


def degrade_one(
        hr: np.ndarray,
        shift_hr: Shift,
        scale: int = 2,
        blur_type: str = "defocus",
        blur_kernel: int = 5,
        blur_sigma: float = 0.0,
        blur_stage: str = "before_downsample",
        border_mode: str = "replicate",
        interpolation: str = "area",
) -> np.ndarray:
    """Forward model: HR -> Blur -> Shift -> Downsample (Golden Standard)"""
    x = to_float01(hr)

    def apply_blur(img_in):
        if blur_type.lower() == "defocus":
            return defocus_blur(img_in, blur_kernel)
        return gaussian_blur(img_in, blur_kernel, blur_sigma)

    # 1. 物理光学衍射/散焦模糊 (发生在真实世界)
    if blur_stage == "before_downsample":
        x = apply_blur(x)

    # 2. 亚像素微小位移
    x = warp_hr(x, shift_hr, border_mode=border_mode)

    # 3. 传感器面积积分与采样
    y = downsample(x, scale=scale, interpolation=interpolation)

    # 理论上不该在采集后模糊，但保留此选项以做特殊实验对比
    if blur_stage == "after_downsample":
        y = apply_blur(y)

    return np.clip(y, 0.0, 1.0)


def degrade_sequence(hr: np.ndarray, cfg: DegradationConfig) -> tuple[list[np.ndarray], list[Shift]]:
    shifts = list(cfg.shifts) if cfg.shifts is not None else [Shift(0.0, 0.0), Shift(0.5, 0.0), Shift(0.0, 0.5),
                                                              Shift(0.5, 0.5)]
    shifts_hr = convert_shifts_to_hr(shifts, cfg.scale, cfg.shift_unit)
    rng = np.random.default_rng(cfg.seed)
    frames: list[np.ndarray] = []

    for sh in shifts_hr:
        lr = degrade_one(
            hr,
            sh,
            scale=cfg.scale,
            blur_type=cfg.blur_type,
            blur_kernel=cfg.blur_kernel,
            blur_sigma=cfg.blur_sigma,
            blur_stage=cfg.blur_stage,
            border_mode=cfg.border_mode,
            interpolation=cfg.interpolation,
        )

        # 4. 模拟电子传感器最后一步：引入泊松-高斯噪声
        lr = add_poisson_gaussian_noise(lr, cfg.noise_poisson_a, cfg.noise_gaussian_b, rng)

        frames.append(lr.astype(np.float32))

    return frames, shifts_hr