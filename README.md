# PixelShiftSR_Mature

一个面向 **像素位移超分辨率** 的可直接运行项目：

1. 将一张高分辨率图像 `HR` 按照亚像素位移、模糊、下采样、噪声退化成四张低分辨率图像 `LR_00 ~ LR_03`；
2. 再利用这四张低分辨率图像重建高分辨率图像 `SR`；
3. 输出 PSNR、SSIM、误差图、对比图和完整 JSON 报告。

本项目重构并工程化了你的原始脚本思想：

```text
HR image -> sub-pixel shift -> downsample -> Gaussian blur -> noise -> four LR frames
four LR frames -> pixel-shift reconstruction -> SR image
```

---

## 1. 安装

```bash
cd PixelShiftSR_Mature
pip install -r requirements.txt
pip install -e .
```

安装后可以使用两种入口：

```bash
python main.py --help
# 或
pssr --help
```

---

## 2. 一键完整 demo：HR -> 四张 LR -> SR

```bash
bash scripts/run_demo.sh
```

输出目录：

```text
runs/demo_pipeline/
├── hr_gt.png                 # 高分辨率真值
├── lr_frames/                # 四张退化后的低分辨率图像
│   ├── lr_00.png
│   ├── lr_01.png
│   ├── lr_02.png
│   └── lr_03.png
├── bicubic_baseline.png      # 单纯双三次/平均上采样基线
├── sr_result.png             # 像素位移超分结果
├── comparison.png            # GT / LR / SR 可视化对比
├── error_map.png             # 误差图
└── pipeline_report.json      # 指标与实验配置
```

---

## 3. 使用你自己的高分辨率图像生成四张低分辨率图像

假设你的高分辨率图像是：

```text
data/hr_gt.png
```

运行：

```bash
python main.py degrade \
  --hr data/hr_gt.png \
  --output runs/my_lr \
  --scale 2 \
  --shifts "0,0;0.5,0;0,0.5;0.5,0.5" \
  --shift-unit lr \
  --blur-kernel 3 \
  --blur-stage after_downsample \
  --noise-poisson-a 0.0 \
  --noise-gaussian-b 0.000984
```

生成：

```text
runs/my_lr/
├── lr_00.png
├── lr_01.png
├── lr_02.png
├── lr_03.png
└── degradation_meta.json
```

---

## 4. 从四张低分辨率图像重建高分辨率图像

```bash
python main.py reconstruct \
  --input runs/my_lr \
  --output runs/my_recon \
  --gt data/hr_gt.png \
  --scale 2 \
  --shifts "0,0;0.5,0;0,0.5;0.5,0.5" \
  --shift-unit lr \
  --mode ibp \
  --iterations 80 \
  --step-size 0.20 \
  --blur-kernel 3 \
  --blur-stage after_downsample \
  --save-debug
```

输出：

```text
runs/my_recon/
├── sr_result.png
├── bicubic_baseline.png
├── comparison.png
├── error_map.png
└── reconstruction_report.json
```

---

## 5. 三种重建模式

### 5.1 `bicubic`

普通上采样基线，只用于对比。

```bash
--mode bicubic
```

### 5.2 `interlace`

已知位移且严格落在 2x HR 网格时使用。理想情况下，四张图分别填入：

```text
HR[0::2, 0::2]
HR[0::2, 1::2]
HR[1::2, 0::2]
HR[1::2, 1::2]
```

```bash
--mode interlace
```

### 5.3 `fuse`

先把每张 LR 上采样到 HR 网格，再根据位移反向对齐和加权平均。
适合位移不是严格整数 HR 网格的情况。

```bash
--mode fuse
```

### 5.4 `ibp` / `inverse`

推荐正式实验使用。

IBP 是 iterative back-projection，即让重建出来的 SR 图像经过同样的退化模型后，尽量重新生成原始四张 LR 图像。

```bash
--mode ibp
```

---

## 6. 关于 `shift-unit` 的重要说明

项目支持两种位移单位。

### 6.1 推荐物理定义：`--shift-unit lr`

```bash
--shifts "0,0;0.5,0;0,0.5;0.5,0.5" --shift-unit lr
```

含义：`0.5` 表示 **半个低分辨率传感器像素**。

当 `scale=2` 时，半个 LR 像素等于 `1` 个 HR 网格像素。这更符合真实像素位移超分：

```text
LR pixel size = 2 HR pixels
0.5 LR pixel = 1 HR pixel
```

### 6.2 兼容你的原始脚本：`--shift-unit hr`

你的原始代码中：

```python
shifts = [(dx/scale, dy/scale) for dx in range(2) for dy in range(2)]
cv2.warpAffine(hr_bayer, M, (W, H))
```

也就是在 HR 图像上直接平移 `0` 或 `0.5` 个 HR 像素。
为了完全复现实验脚本，可以使用：

```bash
bash scripts/run_legacy_like.sh
```

或手动运行：

```bash
python main.py pipeline \
  --output runs/legacy_like_pipeline \
  --shift-unit hr \
  --shifts "0,0;0.5,0;0,0.5;0.5,0.5" \
  --noise-poisson-a 0.0 \
  --noise-gaussian-b 0.000984
```

---

## 7. 针对你们真实系统的推荐命令

你们的实验条件是：

- 相机原始图像：1024 × 1024
- 像素尺寸：5 μm
- 目标重建：2048 × 2048
- 传感器移动：2.5 μm = 0.5 LR pixel
- 四张图：`(0,0), (0.5,0), (0,0.5), (0.5,0.5)` LR-pixel units

推荐使用：

```bash
python main.py reconstruct \
  --input data/real_lr \
  --output runs/real_recon \
  --scale 2 \
  --shifts "0,0;0.5,0;0,0.5;0.5,0.5" \
  --shift-unit lr \
  --mode ibp \
  --iterations 100 \
  --step-size 0.15 \
  --blur-kernel 3 \
  --blur-stage after_downsample \
  --save-debug
```

如果真实图像中噪声明显，可以尝试：

```bash
--tv-weight 0.05
```

如果出现振铃或过锐，可以降低：

```bash
--step-size 0.08
```

---

## 8. 文件结构

```text
PixelShiftSR_Mature/
├── pixelshiftsr/
│   ├── degradation.py        # HR -> shifted LR forward degradation model
│   ├── reconstruction.py     # bicubic / interlace / fuse / IBP reconstruction
│   ├── registration.py       # phase correlation registration utility
│   ├── metrics.py            # PSNR / SSIM / sharpness
│   ├── visualization.py      # comparison and error-map generation
│   ├── synthetic.py          # synthetic HR image and RGGB mask generator
│   ├── io.py                 # image I/O
│   ├── utils.py              # shift parsing, dtype conversion
│   └── cli.py                # command-line interface
├── configs/
│   └── default.yaml
├── scripts/
│   ├── run_demo.sh
│   └── run_legacy_like.sh
├── tests/
│   └── test_smoke.py
├── requirements.txt
├── pyproject.toml
├── main.py
└── README.md
```

---

## 9. 快速检查项目是否可用

```bash
pytest tests
```

也可以直接跑：

```bash
python main.py pipeline --output runs/check --height 128 --width 128 --iterations 5
```

---

## 10. 研究建议

建议你正式实验时保留三组结果：

1. `bicubic_baseline.png`：普通上采样；
2. `fuse`：位移对齐融合；
3. `ibp`：物理退化一致性重建。

论文或汇报中可以表达为：

```text
We first synthesize four sub-pixel shifted LR observations from an HR image using a controlled forward degradation model. Then, the HR image is reconstructed by enforcing consistency between the recovered HR estimate and all LR observations under the same degradation model.
```
