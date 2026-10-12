# AI质检 - 钢铁表面缺陷检测系统

基于 YOLOv12 与多模态大模型 Qwen-VL 双引擎的工业质检系统，将缺陷检测、人工审核、统计分析、Bad Case 导出重训练串成完整的数据闭环，并提供 Gradio 可视化工作台。

## 项目特色

- **双检测引擎**：YOLO 专用检测模型（毫秒级、精确 bbox）+ Qwen-VL 多模态大模型（自然语言缺陷分析 + 定位），统一抽象为 `DetectionResult`，策略模式可插拔、可对比评估
- **数据闭环**：检测 → SQLite 落库 → 人工审核（正确/误检/漏检）→ Bad Case 导出 → 重新标注 → 重训练 → 新模型上线，完整 MLOps 流程
- **竞赛级提分实践**：离线数据增强（数据集扩充 4 倍）、TTA 测试时增强、置信度阈值寻优
- **一键全流程**：训练 + 预测 + 生成提交文件一个脚本完成，自动适配 CPU/GPU

## 实测结果（RTX 4090, 100 epochs, batch 16）

| 指标 | 数值 |
|---|---|
| mAP50 | 0.813 |
| mAP50-95 | 0.453 |
| 最优置信度阈值 | 0.2（阈值寻优后 mAP50 = 0.821） |
| 批量预测速度 | 400 张测试图约 5 秒 |
| 离线增强后数据量 | 1400 张 → 5600 张（每图扩充 3 倍） |

数据集：NEU-DET 钢铁表面缺陷，6 类（crazing 龟裂 / inclusion 夹杂 / pitted_surface 点蚀 / scratches 划痕 / patches 斑块 / rolled-in_scale 氧化铁皮压入），训练集 1400 张，测试集 400 张。

## 项目结构

```
AI质检/
├── app.py                    # Gradio 工作台（5 个 Tab：检测/记录/统计/导出/竞赛提交）
├── detection_engine.py       # 检测引擎封装（YOLO + VLM，含 AAttn 兼容补丁）
├── db_manager.py             # SQLite 持久化（检测记录 + 审核 + 统计）
├── data_export.py            # Bad Case / 重训练数据导出
├── train_and_predict.py      # 训练 + 预测 + 生成 submission.csv（一个脚本全流程）
├── train_yolo_enhanced.py    # 增强训练策略集（baseline/enhanced/augment/tta/conf_tune）
├── generate_submission.py    # 独立批量预测生成提交文件
├── eda_analysis.py           # 数据集 EDA 分析（类别分布/框分析/可视化图表）
├── augmentation_demo.py      # 数据增强效果可视化
├── test_vlm_detect.py        # VLM 检测独立测试脚本
├── requirements.txt
├── models/best.pt            # 训练好的模型（app.py 优先加载）
├── inspection.db             # SQLite 检测记录数据库
├── yolo-cases/
│   ├── steel_data/           # NEU-DET 数据集（train/images, train/labels, test/images）
│   └── yolov12.yaml          # YOLOv12 模型结构配置
├── eda_output/               # EDA 分析图表输出
├── augmentation_output/      # 增强可视化输出
└── runs/steel_train/         # 训练输出（weights/best.pt）
```

## 环境安装

```bash
# Python 3.11 / 3.12
pip install -r requirements.txt
```

核心依赖：torch 2.5.1、ultralytics 8.3.100、gradio、openai（VLM 调用）、albumentations、opencv、pandas、matplotlib。

> 注：requirements 原始钉的 torch 2.7.0 / numpy 2.4.3 存在相互冲突，已调整为实测通过的 torch 2.5.1 + numpy 1.26.4 组合。

VLM 检测需要阿里云 DashScope API Key（模型 qwen3-vl-plus）：

```bash
# Linux / macOS
export DASHSCOPE_API_KEY=你的Key
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "你的Key"
```

不配置也能正常使用 YOLO 检测，仅 VLM 功能不可用。

## 快速开始

### 1. 数据集 EDA 分析

```bash
python eda_analysis.py          # 输出图表到 eda_output/
python augmentation_demo.py     # 增强效果可视化到 augmentation_output/
```

### 2. 训练 + 预测（一个脚本全流程）

```bash
# GPU 正式训练（默认 100 epochs, batch 16），训练完自动预测生成 submission.csv
python train_and_predict.py --device 0 --epochs 100 --batch 16

# 增强数据增强版（推荐竞赛使用）
python train_and_predict.py --device 0 --epochs 150 --batch 16 --enhanced

# CPU 快速验证流程（1 个 epoch）
python train_and_predict.py

# 只用已训练好的模型生成 submission.csv
python train_and_predict.py --predict_only --model ./models/best.pt

# 预测 + TTA（测试时增强，可提分）
python train_and_predict.py --predict_only --model ./models/best.pt --tta
```

模型保存在 `runs/steel_train/weights/best.pt`，将其复制到 `models/` 目录供工作台加载。

### 3. 增强训练策略

```bash
python train_yolo_enhanced.py --strategy tips        # 竞赛提分技巧清单
python train_yolo_enhanced.py --strategy augment     # 离线数据增强（扩充3倍）
python train_yolo_enhanced.py --strategy tta --model_path runs/steel_train/weights/best.pt
python train_yolo_enhanced.py --strategy conf_tune --model_path runs/steel_train/weights/best.pt  # 置信度阈值寻优
```

### 4. 启动工作台

```bash
python app.py    # 默认 http://127.0.0.1:7860
```

工作台包含 5 个 Tab：

| Tab | 功能 |
|---|---|
| 缺陷检测 | 上传图片或从测试集选图，选 YOLO/VLM 引擎检测，现场人工审核 |
| 检测记录 | 历史记录筛选、详情回看（含原图）、补审核 |
| 统计分析 | 检测总数、模型分布、审核准确率、缺陷类别分布、平均耗时 |
| 数据导出 | 导出误检/漏检 Bad Case 与重训练数据包 |
| 竞赛提交 | 批量预测 400 张测试集，可调 conf/IoU/TTA，生成并下载 submission.csv |

### 5. VLM 检测单独测试

```bash
python test_vlm_detect.py --image yolo-cases/steel_data/train/images/0.jpg
```

## AutoDL 部署要点

在 AutoDL 等国内 GPU 服务器部署时，以下问题已解决，可参考：

1. **依赖冲突**：同时钉 PyYAML 两个版本、numpy 2.x 与 torchvision 0.20.1 不兼容，按上文版本组合安装
2. **AAttn 权重兼容**：从 `yolov12.yaml` 训出的权重是 qk+v 分离格式，pip 版 ultralytics 加载会报 `'AAttn' object has no attribute 'qkv'`。`detection_engine.py` 中的 `_patch_aattn_forward()` 已做前向兼容；`generate_submission.py`、`train_yolo_enhanced.py` 通过 `import detection_engine` 自动应用补丁
3. **中文字体**：Linux 无 SimHei，可视化脚本已添加 `Noto Sans CJK SC` fallback，需 `apt-get install -y fonts-noto-cjk`
4. **被墙资源**：ultralytics 首次运行会下载 Arial.ttf 和 AMP 测试图（bus.jpg/zidane.jpg），国内不通。解决：手动放置占位文件到 `~/.config/Ultralytics/Arial.ttf`，并用任意本地 jpg 顶替 `site-packages/ultralytics/assets/bus.jpg` 与 `zidane.jpg`
5. **Gradio 端口**：AutoDL 自定义服务固定代理 6006 端口，将 `app.launch` 中 `server_port` 改为 6006 即可通过控制台给出的链接公网访问；本地访问也可用 SSH 隧道 `ssh -N -L 7860:127.0.0.1:6006 <实例>`
6. **数据集路径**：`steel_data/dataset.yaml` 中 `path` 需改为当前机器的实际路径（或直接使用 `train_and_predict.py` 自动生成的 `dataset_local.yaml`）

## 常见问题

- **检测时提示模型文件不存在**：确认 `models/best.pt` 或 `runs/steel_train/weights/best.pt` 至少存在一个
- **记录时间慢 8 小时**：SQLite 的 `CURRENT_TIMESTAMP` 为 UTC 时间，属正常现象
- **YOLO 与 VLM 如何选择**：产线高频检测用 YOLO（免费、毫秒级）；疑难图片复核、缺陷原因分析用 VLM（秒级、按调用计费）
- **如何参与模型迭代**：在工作台检测 → 审核标记误检/漏检 → Tab4 导出 Bad Case → 重新标注后并入训练集 → 重新执行训练流程
