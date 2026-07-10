# Examples

This folder is intentionally empty. Use:

```bash
python main.py make-demo --output examples/demo_hr --height 512 --width 512
python main.py degrade --hr examples/demo_hr/hr_gt.png --output examples/lr_frames --noise-poisson-a 0.0 --noise-gaussian-b 0.0000615
python main.py reconstruct --input examples/lr_frames --output examples/recon --gt examples/demo_hr/hr_gt.png --mode ibp --iterations 80 --save-debug
```
