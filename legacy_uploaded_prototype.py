import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation

# --------------------------
# 配置参数 可自行修改
# --------------------------
SCALE = 2               # 超分放大倍数，适配拜耳2x2周期
NUM_FRAMES = 4          # 拍4帧微位移图像，经典配置
BLUR_KERNEL = 3         # 模拟实拍镜头模糊
NOISE_LEVEL = 8         # 模拟传感器噪声

# --------------------------
# 1. 生成标准清晰 RGGB 拜耳阵列
# --------------------------
def generate_bayer_pattern(H, W):
    bayer = np.zeros((H, W, 3), dtype=np.float32)
    # RGGB 排布
    for y in range(H):
        for x in range(W):
            # 偶数行
            if y % 2 == 0:
                if x % 2 == 0:
                    bayer[y,x,0] = 1.0  # R
                else:
                    bayer[y,x,1] = 1.0  # G
            # 奇数行
            else:
                if x % 2 == 0:
                    bayer[y,x,1] = 1.0  # G
                else:
                    bayer[y,x,2] = 1.0  # B
    return (bayer * 255).astype(np.uint8)

# --------------------------
# 2. 生成多帧 亚像素位移 + 模糊 拜耳图像
# --------------------------
def make_shifted_bayer_seq(hr_bayer, scale, num_frames):
    H, W = hr_bayer.shape[:2]
    lr_H, lr_W = H // scale, W // scale
    frame_list = []
    # 生成2x2规则亚像素位移
    shifts = [(dx/scale, dy/scale) for dx in range(2) for dy in range(2)]
    
    for dx, dy in shifts[:num_frames]:
        # 亚像素平移
        M = np.array([[1, 0, dx],
                      [0, 1, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(hr_bayer, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        # 下采样模拟低分辨率实拍
        lr = cv2.resize(shifted, (lr_W, lr_H), interpolation=cv2.INTER_AREA)
        # 高斯模糊 + 加噪声 模拟真实拍照
        lr = cv2.GaussianBlur(lr, (BLUR_KERNEL, BLUR_KERNEL), 0)
        noise = np.random.normal(0, NOISE_LEVEL, lr.shape).astype(np.float32)
        lr = np.clip(lr + noise, 0, 255).astype(np.uint8)
        frame_list.append(lr)
    return frame_list, shifts

# --------------------------
# 3. 像素位移 多帧超分重构 拜耳阵列
# --------------------------
def bayer_subpixel_sr(frame_seq, scale):
    lr_H, lr_W = frame_seq[0].shape[:2]
    hr_H, hr_W = lr_H * scale, lr_W * scale
    
    hr_acc = np.zeros((hr_H, hr_W, 3), dtype=np.float32)
    count  = np.zeros((hr_H, hr_W, 3), dtype=np.float32)
    
    for frm in frame_seq:
        # 单帧上采样到高分辨网格
        up = cv2.resize(frm, (hr_W, hr_H), interpolation=cv2.INTER_LINEAR)
        hr_acc += up
        count += 1
    
    # 平均去噪
    hr_out = hr_acc / count
    return np.clip(hr_out, 0, 255).astype(np.uint8)

# --------------------------
# 主流程
# --------------------------
if __name__ == "__main__":
    # 生成高分辨标准拜耳
    H, W = 128, 128
    hr_bayer_gt = generate_bayer_pattern(H, W)
    
    # 生成多帧微位移低清拜耳实拍序列
    frame_seq, shifts = make_shifted_bayer_seq(hr_bayer_gt, SCALE, NUM_FRAMES)
    
    # 像素位移超分重构
    sr_bayer = bayer_subpixel_sr(frame_seq, SCALE)
    
    # 取第一帧低清原图做对比
    raw_low = cv2.resize(frame_seq[0], (W, H), interpolation=cv2.INTER_NEAREST)
    
    # 可视化对比
    plt.figure(figsize=(12,5))
    plt.subplot(131)
    plt.imshow(hr_bayer_gt)
    plt.title("标准清晰拜耳阵列")
    plt.axis("off")
    
    plt.subplot(132)
    plt.imshow(raw_low)
    plt.title("单张实拍模糊拜耳")
    plt.axis("off")
    
    plt.subplot(133)
    plt.imshow(sr_bayer)
    plt.title("像素位移超分 还原拜耳")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
