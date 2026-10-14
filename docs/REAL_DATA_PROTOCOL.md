# 真实数据采集与验收规范

本文档用于把算法实验转换为可重复的硬件实验。正式专利实施例应记录实际设备型号、参数范围和原始数据，不应只使用模拟图。

## 1. 最低硬件条件

- 固定焦距相机或工业相机，关闭自动曝光、自动增益、自动白平衡和自动锐化；
- 二维微位移平台、压电执行器或带编码器的相机/传感器位移机构；
- 刚性支架、稳定照明和遮光环境；
- 可重复显示或印刷的标定靶；
- 能记录曝光时间、增益、温度、平台指令和实际编码器值的采集程序。

## 2. 建议采集顺序

2 倍超分的基础序列使用 LR 像素单位：

    frame_00: (0.0, 0.0)
    frame_01: (0.5, 0.0)
    frame_02: (0.0, 0.5)
    frame_03: (0.5, 0.5)

每次位移后应等待机械振动衰减，再触发曝光。建议额外采集冗余帧，使系统能够在剔除异常帧后仍保留至少四个有效观测。

## 3. 标定数据

每套光学和曝光设置至少保存：

- 暗场：遮光后同曝光参数采集 16 帧以上；
- 平场：均匀照明下采集 16 帧以上；
- PSF/边缘靶：已知 HR 标定图及对应位移 LR 序列；
- 几何标定：畸变标定板；
- 重复性：相同位移循环至少 10 次，统计均值和标准差。

更换镜头、焦距、光圈、工作距离、曝光、增益或温度范围后，应重新评估标定是否有效。

## 4. 数据目录

    data/experiment_001/
    |-- lr/
    |   |-- frame_00.png
    |   |-- frame_01.png
    |   |-- frame_02.png
    |   `-- frame_03.png
    |-- calibration/
    |-- metadata.json
    `-- notes.md

metadata.json 至少记录相机、镜头、平台、曝光和实际位移：

    {
      "camera_model": "填写",
      "sensor_pixel_um": 0.0,
      "lens_model": "填写",
      "working_distance_mm": 0.0,
      "exposure_us": 0,
      "gain_db": 0.0,
      "stage_model": "填写",
      "commanded_shifts_lr": [[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]],
      "encoder_shifts_um": [],
      "temperature_c": 0.0
    }

## 5. 标定与重建

先标定：

    python main.py calibrate ^
      --hr data/calibration/hr_target.png ^
      --input data/calibration/lr ^
      --output runs/calibration ^
      --registration estimate

再把最佳 PSF 和噪声参数写入冻结配置，并重建真实序列：

    python main.py reconstruct ^
      --config configs/experiment_001.yaml ^
      --input data/experiment_001/lr ^
      --output runs/experiment_001 ^
      --registration estimate ^
      --save-debug

PowerShell 用户可直接运行 scripts/run_real.ps1。

## 6. 建议验收门槛

以下值是实验起点，需根据设备能力重新确定：

- 有效帧数不少于 4；
- 自动配准置信度建议不低于 0.6；
- 实际位移与目标位移的绝对误差建议不超过 0.08 LR 像素；
- 饱和像素比例低于 0.5%；
- SR 重投影 MSE 低于双三次基线；
- 多次重复实验的核心指标给出均值、标准差和最差值；
- 最终工业任务同时报告检出率、误报率、尺寸误差或 MTF，而非只报告清晰度。

## 7. 专利实验注意事项

- 保留原始未处理图、标定图、配置、报告和采集日志；
- 结果图必须能追溯到原始文件哈希和软件版本；
- 对比至少包含单帧双三次、位移融合、普通 L2 IBP 和本项目鲁棒方法；
- 分别消融自动配准、置信度筛帧、PSF 标定和 Huber 损失；
- 在提交申请前确认公开演示、论文、仓库和销售行为是否会影响新颖性。
