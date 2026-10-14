# PixelShiftSR 2.0

面向像素位移超分辨率实验与工业成像验证的完整工程。项目覆盖物理退化、亚像素配准、置信度筛帧、鲁棒迭代反投影、模型标定、无真值评价和可追溯报告，可用于形成专利技术交底所需的实现与实验依据。

> 本项目能提供技术实现和实验材料，但软件本身不等于已经具备专利授权条件。申请前仍需检索现有技术，并由专利代理师结合真实硬件方案确定权利要求范围。

## 核心能力

| 模块 | 能力 |
| --- | --- |
| 成像模型 | 光学 PSF、亚像素位移、像素积分、泊松-高斯噪声 |
| 自动配准 | 相位相关粗配准、ECC 精配准、置信度评分、异常帧剔除 |
| 超分重建 | 双三次、直接插值、位移融合、置信度加权鲁棒 IBP |
| 模型标定 | 从已知 HR 标定靶和 LR 实拍序列搜索 PSF，并拟合噪声参数 |
| 质量评价 | PSNR、SSIM、RMSE、清晰度、Tenengrad、重投影一致性 |
| 证据留存 | 输入 SHA-256、完整参数、依赖版本、逐帧指标、收敛曲线 |

统一位移约定：正 `dx/dy` 表示图像内容向右/向下移动。报告同时保存前向位移和对齐位移，避免符号混淆。

## 安装

要求 Python 3.9 或更高版本。

```bash
python -m pip install -e ".[dev]"
python main.py --help
```

安装后也可以使用命令 `pssr` 代替 `python main.py`。

## 一键验收

Windows PowerShell：

```powershell
.\scripts\run_demo.ps1
```

Linux 或 macOS：

```bash
bash scripts/run_demo.sh
```

也可以直接运行小型实验：

```bash
python main.py pipeline \
  --config configs/default.yaml \
  --output runs/demo \
  --preset default \
  --height 256 --width 256 \
  --iterations 40 --no-progress
```

主要输出：

```text
runs/demo/
|-- hr_gt.png
|-- lr_frames/
|   |-- lr_00.png ... lr_03.png
|   `-- degradation_meta.json
|-- sr_result.png
|-- sr_result.npy
|-- bicubic_baseline.png
|-- comparison.png
|-- convergence.png
|-- error_map.png
`-- pipeline_report.json
```

PNG 默认保存为 16 位，`sr_result.npy` 保存未经量化的 `float32` 重建结果。

## 重建真实序列

将同一场景的连续采集帧放入一个目录，例如 `data/real_lr/`。建议锁定曝光、增益、焦距和白平衡，并确保平台每次位移后充分稳定。

```powershell
.\scripts\run_real.ps1 -InputDirectory data\real_lr
```

等价命令：

```bash
python main.py reconstruct \
  --config configs/default.yaml \
  --input data/real_lr \
  --input-pattern "*.png" \
  --output runs/real_reconstruction \
  --registration estimate \
  --save-debug
```

自动配准会在报告中记录每帧的实际位移、相关系数、置信度、重投影误差和是否被采用。若位移由高精度闭环平台直接提供，可使用 `--registration known` 和 `--shifts`。

## 标定成像模型

准备一张已知高分辨率标定靶图像和对应的低分辨率采集序列：

```bash
python main.py calibrate \
  --hr data/calibration/hr_target.png \
  --input data/calibration/lr_frames \
  --input-pattern "*.png" \
  --output runs/calibration \
  --registration estimate \
  --blur-types defocus,gaussian \
  --blur-kernels 1,3,5,7 \
  --blur-sigmas 0.6,1.0,1.4
```

`calibration_report.json` 会给出最佳 PSF 参数、候选模型误差排序，以及 `variance = a * signal + b` 的噪声拟合结果。正式实验应把标定值写回独立配置文件并冻结。

## 对比与消融实验

专利技术效果不应只展示单一结果。`benchmark` 会在同一组输入和位移上依次运行双三次、位移融合、L2 IBP 和 Huber IBP，并按 PSNR 或无真值重投影误差排序：

```bash
python main.py benchmark \
  --config configs/default.yaml \
  --input data/real_lr \
  --input-pattern "*.png" \
  --output runs/benchmark \
  --registration estimate \
  --variants bicubic,fuse,ibp_l2,ibp_huber
```

提供 `--gt` 时按 PSNR 排序；没有真值时按加权重投影 MSE 排序。输出包括每种方法的 16 位 PNG、float32 NPY、收敛曲线、耗时、完整指标和 `benchmark_report.json`。

## 配置文件

[`configs/default.yaml`](configs/default.yaml) 已经可以被程序直接读取，不再只是参数说明。配置文件包含：

- `model`：光学模糊、采样和边界模型；
- `degradation`：噪声、随机种子和输出位深；
- `registration`：粗精配准、阈值和筛帧规则；
- `reconstruction`：鲁棒损失、迭代、收敛和正则化；
- `calibration`：PSF 搜索空间。

真实设备可从 [`configs/real_template.yaml`](configs/real_template.yaml) 复制一份独立配置，再写入本机标定结果。

命令行显式参数优先于 YAML，例如：

```bash
python main.py pipeline --config configs/default.yaml --iterations 20
```

## 算法定义

第 `k` 帧的成像过程表示为：

```text
y_k = D(B(T(s_k)x)) + n_k
```

其中 `T` 为亚像素平移，`B` 为标定 PSF，`D` 为像素积分和降采样，`n_k` 为信号相关噪声。重建采用置信度加权的鲁棒数据一致性目标：

```text
min_x sum_k w_k * Huber(y_k - D(B(T(s_k)x))) + regularization(x)
```

反投影包含降采样、位移和对称 PSF 的近似伴随算子，并使用覆盖度归一化。该实现与旧版“残差直接放大再平均”不同。

## 验证结果

固定随机种子、128 x 128 通用测试图、4 帧、2 倍重建、40 次迭代：

| 条件 | SR PSNR | 双三次 PSNR | 增益 | 重投影 MSE 降幅 |
| --- | ---: | ---: | ---: | ---: |
| 无模糊、无噪声 | 31.41 dB | 22.39 dB | +9.02 dB | 99.46% |
| 3 像素散焦、泊松-高斯噪声 | 26.54 dB | 21.94 dB | +4.60 dB | 93.54% |

自动配准验收中，目标半像素位移被估计为约 `0.51` 至 `0.53` LR 像素，四帧均通过置信度筛选。模型标定验收正确找回 `defocus, kernel=3`，噪声拟合结果接近生成参数。

## 测试

```bash
python -m pytest -q
```

测试覆盖前向位移、噪声校验、位移符号、无纹理拒绝、重建增益、重投影一致性、PSF 标定、对比实验、YAML 优先级和严格 JSON 输出。GitHub Actions 会在 Python 3.9 至 3.12 上执行同一套测试。

## 工程结构

```text
pixelshiftsr/
|-- degradation.py       # 成像前向模型及伴随反投影
|-- registration.py      # 粗配准、ECC 精配准和置信度
|-- reconstruction.py    # 融合和鲁棒 IBP
|-- calibration.py       # PSF 与噪声标定
|-- metrics.py           # 有真值和无真值指标
|-- reporting.py         # 可追溯、严格 JSON 报告
|-- config.py            # YAML 配置加载和 CLI 覆盖
`-- cli.py               # 命令入口
```

进一步材料：

- [真实数据采集与验收规范](docs/REAL_DATA_PROTOCOL.md)
- [系统架构说明](docs/ARCHITECTURE.md)
- [专利技术交底书草案](docs/TECHNICAL_DISCLOSURE.md)

## 当前边界

- 当前运动模型为全局二维平移，存在旋转、透视、局部形变时需要扩展；
- PSF 当前为圆盘或高斯核，真实镜头可进一步接入实测非对称核；
- Bayer RAW、坏点、暗场、平场和镜头畸变尚未进入主链路；
- 无真值清晰度指标只能作为辅助，最终工业价值应由缺陷检出率或尺寸测量误差验证。
