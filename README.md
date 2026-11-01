# YOLO 试管检测

生物试剂盒 + 机械臂抓取项目的识别端：用 YOLO 检测试剂盒托盘中插了试管的孔位，输出可供机械臂按格抓取的中心点坐标。

取代旧版 `cs_腐蚀_3_最终版.py`（传统腐蚀 + 距离变换 + Hough 圆）流水线，不再手工标定阈值/半径，托盘颜色、光照变化由训练数据统一学习。

## 文件说明

| 文件 | 说明 |
|---|---|
| `cs_试管_yolo检测.py` | 唯一入口脚本：import-roboflow / validate / train / predict / export |
| `yolo11n.pt` | 预训练基础模型，作为 `--model` 训练起点 |
| `tube_dataset/` | 标准数据集目录（`images/{train,val}`、`labels/{train,val}`） |
| `tube_roboflow/` | 已导入的 Roboflow 数据（2 类）+ 已训练权重 `runs/train/weights/best.pt` |
| `tube.v1i.yolov11/` | Roboflow 原始导出（train/valid/test），供 `import-roboflow` 使用 |

## 环境依赖

- Python 3.x
- `pip install ultralytics opencv-python`
- Windows 下默认 `--workers 0`、`--device cpu`；有 CUDA 可用 `--device 0`

## 快速开始

所有操作都通过同一个脚本（PowerShell 中文件名加引号）：

```powershell
# 1. 导入 Roboflow 数据（zip 或解压后的文件夹均可）
python "cs_试管_yolo检测.py" --mode import-roboflow --roboflow tube.v1i.yolov11 --dataset tube_dataset

# 2. 校验数据集（检查图片/标签配对、标签格式）
python "cs_试管_yolo检测.py" --mode validate --dataset tube_dataset

# 3. 训练（权重输出到 <dataset>/runs/train/weights/best.pt）
python "cs_试管_yolo检测.py" --mode train --dataset tube_dataset

# 4. 推理（--input 可为单图或目录，自动递归收集）
python "cs_试管_yolo检测.py" --mode predict --input <图片或目录> --weights tube_dataset/runs/train/weights/best.pt

# 5. 导出 ONNX
python "cs_试管_yolo检测.py" --mode export --weights tube_dataset/runs/train/weights/best.pt
```

`predict` 可选参数：

- `--use-roi`：只检测中央 90%×86% ROI（与旧算法一致），减少画面边缘误检
- `--compare`：动态加载旧腐蚀算法叠加对比（需要 `cs_腐蚀_3_最终版.py` 在脚本同目录）
- `--conf` / `--iou` / `--max-det`：置信度、NMS 阈值、最大检测数

预训练权重路径：`tube_dataset/runs/train/weights/best.pt`

## 输出格式（推断）

每张图输出到 `--output`（默认 `检测结果yolo/`）：

- `<图名>_yolo标注.png`：标注图（绿框 + 红色十字中心点 + `id 置信度`）
- `<图名>_yolo坐标.csv`：UTF-8 BOM，Excel 可直接打开
- `<图名>_yolo坐标.json`：完整检测结果（含 ROI、逐框坐标/半径/置信度/中心点）
- `yolo汇总.json`：所有图片数量汇总

字段：`id, x1, y1, x2, y2, x, y, radius, confidence`，坐标为原图像素坐标。

> **重要契约**：检测结果按 y 再 x 排序后分配连续 `id`，机械臂依赖此顺序按格位抓取，请勿改动排序逻辑。

## 进程内接口

供其他模块 import（为改写 ROS 预留）：

```python
from cs_试管_yolo检测 import detect_centers

points = detect_centers("某张图.jpg", "runs/train/weights/best.pt")
# 单图 -> [{"id": 1, "cx": 312.5, "cy": 145.2}, ...]
# 目录 -> [{"image": 路径, "count": n, "points": [...]}, ...]
```

## 数据集说明

- 标注格式：`class_id cx cy w h`（归一化 0~1），只标注插了试管的孔位，空孔和加强筋不标
- `import-roboflow` 会把 train/valid/test 统一为 train/val（valid、test 并入 val），并自动把 `data.yaml` 的 `path` 改写为绝对路径

## 常见问题

- **训练/验证提示找不到图片**：检查 `data.yaml` 的 `path:` 是否为当前机器上的绝对路径（脚本会在 import/validate/train 时自动重写；搬运数据集目录后路径可能失效）
- **`--compare` 报错找不到旧算法**：`cs_腐蚀_3_最终版.py` 不在本目录时无法对比
- **`--input` 默认值指向本机路径** `C:\Users\BlueLR\Pictures\OrangePi`，调用时请始终显式传 `--input`
