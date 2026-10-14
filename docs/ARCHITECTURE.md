# 系统架构

## 处理链路

    采集/模拟 LR 序列
            |
            v
    输入校验与 SHA-256 清单
            |
            v
    相位相关粗配准 -> ECC 亚像素精配准
            |
            v
    置信度评分 -> 异常帧剔除/降权
            |
            v
    标定前向模型 T(位移) + B(PSF) + D(积分采样)
            |
            v
    置信度加权鲁棒 IBP
            |
            +--> 16 位 PNG / float32 NPY
            |
            +--> 重投影、一致性、清晰度、GT 指标
            |
            `--> 严格 JSON 报告与调试图

## 坐标与位移约定

整个工程只使用一种前向位移定义：

    dx > 0: 图像内容向右移动
    dy > 0: 图像内容向下移动

相位相关计算的是“将 moving 对齐到 reference”的位移，成像模型使用的是相反方向的前向位移。RegistrationResult 同时保存：

- alignment_shift_lr：对齐变换；
- forward_shift_lr：成像与重建使用的物理前向位移。

所有报告均写明位移单位。lr 表示传感器低分辨率像素，hr 表示重建网格像素。

## 前向与反投影

前向算子位于 pixelshiftsr/degradation.py：

    HR -> PSF -> forward shift -> pixel integration/downsample -> LR

反投影按相反顺序近似执行：

    LR residual -> sampling adjoint -> inverse shift -> symmetric PSF adjoint -> HR correction

面积采样的伴随算子使用块复制并除以 scale 的平方。多帧校正量再除以伴随覆盖度，抑制边缘和欠采样区域的步长放大。

## 鲁棒优化

pixelshiftsr/reconstruction.py 支持 L2 和 Huber 数据项。默认 Huber 截断像素级异常值，同时将配准置信度作为帧级权重：

    weight(frame k) * influence(residual k)

优化保留最佳迭代结果，并记录完整损失、最佳迭代、停止原因、逐帧重投影 MSE 和覆盖度。

## 标定

pixelshiftsr/calibration.py 使用已知 HR 靶图和对应 LR 序列：

1. 对候选 PSF 类型、核尺寸和高斯 sigma 逐一前向投影；
2. 按多帧平均 MSE 选择最佳模型；
3. 对最佳模型残差按信号区间分箱；
4. 拟合 variance = poisson_a * signal + gaussian_b；
5. 输出前十个候选和最佳模型预测帧。

## 报告契约

报告由 pixelshiftsr/reporting.py 生成，包含：

- UTC 时间、软件和依赖版本、完整命令；
- 所有输入文件的绝对路径、字节数和 SHA-256；
- 解析后的最终配置及 YAML/CLI 覆盖关系；
- 配准、重建、质量和重投影指标；
- 输出文件路径。

非有限浮点数会转换为 JSON null，报告使用 allow_nan=False 写入，保证可被标准 JSON 工具读取。

## 扩展边界

后续模块应保持现有数据契约，可独立扩展：

- 将全局平移替换为仿射、单应或局部光流；
- 将圆盘/高斯 PSF 替换为实测二维非对称核；
- 在前向模型中加入畸变、暗场、平场、Bayer CFA 和坏点；
- 在重建后接入 LED 边缘、亮暗点或线宽检测；
- 将平台编码器位移和曝光元数据写入输入清单。
