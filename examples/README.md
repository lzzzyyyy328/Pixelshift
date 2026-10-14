# 示例

生成已知真值并执行完整实验：

    python main.py pipeline --config configs/default.yaml --output runs/example --preset default --height 256 --width 256 --iterations 40

分步生成低分辨率序列：

    python main.py make-demo --output runs/example_target --preset default
    python main.py degrade --config configs/default.yaml --hr runs/example_target/hr_gt.png --output runs/example_lr

使用已知位移重建：

    python main.py reconstruct --config configs/default.yaml --input runs/example_lr --input-pattern "lr_*.png" --output runs/example_reconstruction --gt runs/example_target/hr_gt.png

使用图像内容估计真实位移：

    python main.py reconstruct --config configs/default.yaml --input data/real_lr --output runs/real_reconstruction --registration estimate --save-debug

输出目录下的 JSON 报告保存最终参数、输入哈希、配准置信度、逐帧重投影误差和质量指标。
