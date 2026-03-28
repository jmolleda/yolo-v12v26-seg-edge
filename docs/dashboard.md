# Results Dashboard

Interactive dashboard with training curves, inference benchmarks, and hardware metrics across all devices.

<a href="/yolo-v12v26-seg-edge/results_dashboard.html" target="_blank" style="display:inline-block;padding:12px 24px;background:#2196F3;color:white;border-radius:6px;text-decoration:none;font-weight:bold;margin:16px 0;">Open Results Dashboard</a>

The dashboard is a self-contained HTML page with Chart.js visualizations covering:

- **Overview** — Training summary table, mAP bar chart, time-vs-accuracy scatter
- **Training Curves** — Epoch-by-epoch loss and mAP with device/experiment/arch/size filters
- **Convergence** — Best epoch, patience usage, seconds per epoch, efficiency
- **Inference** — FPS comparison, stacked latency breakdown across devices
- **Head-to-Head** — v26 vs v12, scratch vs pretrained, AutoBatch comparison
- **Resources** — Parameters, GFLOPs, GPU memory, model file sizes
