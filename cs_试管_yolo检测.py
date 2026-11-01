# -*- coding: utf-8 -*-
"""YOLO 试管检测：基于腐蚀算法经验规划的新方案。

== 设计思路（对应 cs_腐蚀_3_最终版.py 的传统流水线） ==

传统腐蚀算法流水线：
  灰度化 -> 中央 ROI -> CLAHE 增强 -> Otsu/自适应阈值掩膜
  -> 形态学开闭 + 腐蚀去噪 -> 距离变换生成圆心候选
  -> Hough 圆补足 -> 局部环形对比度/中心亮度/饱和度过滤
  -> 圆心 NMS -> 按 y 再 x 排序输出像素坐标

痛点：
  1. 阈值（对比度>=7、中心亮度 40~210、饱和度<190）全部手工标定，
     换托盘颜色、光照或液体颜色就要重调；
  2. 白/蓝/黄托盘各写了一条专用分支，逻辑分散难维护；
  3. 半径范围 (14~48 px) 与拍摄距离强耦合，换分辨率需重新标定。

YOLO 方案如何取代各阶段：
  | 传统阶段                    | YOLO 替代                              |
  |-----------------------------|----------------------------------------|
  | 阈值掩膜 + 腐蚀             | Backbone 自动学习边缘/纹理特征         |
  | 距离变换候选圆心            | Detection Head 直接回归边界框          |
  | 环形对比度区分试管/空孔     | 分类分支学习"试管 vs 空孔"差异         |
  | 多托盘颜色分支              | 多样化训练数据统一学习                 |
  | 手写圆心 NMS                | YOLO 内置 NMS（IoU 阈值可调）          |

保持不变的部分：
  1. 输出格式与旧算法兼容：原图像素坐标 (x, y) + 半径估计 r，
     按 y 再 x 排序，方便机械臂按格抓取；
  2. 推理时可选 ROI 裁剪 (--use-roi)，与旧算法的中央 90%x86% 一致，
     减少画面边缘误检；默认全图检测，兼容网络下载的多样测试图。

数据集准备（从网络下载图片并标注后，放入 tube_dataset）：
  tube_dataset/
    data.yaml              # validate 模式自动生成
    images/train/*.jpg     # 约 80% 训练图
    images/val/*.jpg       # 约 20% 验证图
    labels/train/*.txt     # YOLO 格式: class_id cx cy w h（归一化 0~1）
    labels/val/*.txt

标注建议（下载图片后）：
  1. 只标注插了试管的孔位（含管帽/液面），空孔和加强筋不标；
  2. 框紧贴试管外轮廓，不要留大边距；
  3. 遮挡超过一半的试管也标出可见部分；
  4. 建议 >=200 张，覆盖不同光照/角度/托盘颜色，网络图与实拍图混合。

使用流程（Roboflow Universe 下载后）：
  1. 在 Roboflow Universe 找到目标数据集，选择 "YOLOv8/YOLOv11" 格式下载 zip
  2. python cs_试管_yolo检测.py --mode import-roboflow --roboflow 下载的.zip --dataset tube_dataset
     （或直接传解压后的文件夹；也可用 --dataset tube_roboflow 放独立目录，避免和现有数据混）
  3. python cs_试管_yolo检测.py --mode validate --dataset tube_dataset
  4. python cs_试管_yolo检测.py --mode train --dataset tube_dataset
  5. python cs_试管_yolo检测.py --mode predict --input 某张图片.jpg \
         --weights tube_dataset/runs/train/weights/best.pt
     （--input 也可以是整个文件夹；加 --use-roi 裁剪中央区域，--compare 叠加旧算法对比）
  6. python cs_试管_yolo检测.py --mode export --weights tube_dataset/runs/train/weights/best.pt

进程内接口（其他模块 import 使用，为后续改写 ROS 预留）：
  from cs_试管_yolo检测 import detect_centers
  points = detect_centers("某张图.jpg", "runs/train/weights/best.pt")
  # -> [{"id": 1, "cx": 312.5, "cy": 145.2}, ...]
  # 有几个框返回几个点；id 按 y 再 x 连续编号（机械臂按格抓取契约）；
  # 坐标为原图像素坐标；改写 ROS 时返回值可直接作为消息数据源。
"""

import argparse
import csv
import json
import shutil
import tempfile
import sys
from pathlib import Path

import cv2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_DATASET = "tube_dataset"
DEFAULT_INPUT = r"C:\Users\BlueLR\Pictures\OrangePi"
LEGACY_SCRIPT = "cs_腐蚀_3_最终版.py"


def collect_images(path):
    """递归遍历目录（含子文件夹），收集所有后缀匹配的图像文件。"""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        images = []
        seen = set()
        for x in sorted(p.rglob("*")):
            if not x.is_file() or x.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            resolved = str(x.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            images.append(x)
        return images
    raise FileNotFoundError(str(p))


def dataset_dirs(dataset):
    root = Path(dataset)
    return {
        "root": root,
        "train_images": root / "images" / "train",
        "val_images": root / "images" / "val",
        "train_labels": root / "labels" / "train",
        "val_labels": root / "labels" / "val",
    }


def write_data_yaml(dataset):
    root = Path(dataset).resolve()
    text = (
        "path: {}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: [tube]\n".format(str(root).replace("\\", "/"))
    )
    (root / "data.yaml").write_text(text, encoding="utf-8")
    print("已生成 data.yaml: {}".format(root / "data.yaml"))


def parse_data_yaml_text(text):
    """宽松解析 Roboflow 导出的 data.yaml，返回 (nc, names)。"""
    nc = None
    names = None
    in_names_block = False
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("names:"):
            val = s[len("names:"):].strip()
            if val.startswith("["):
                inner = val.strip("[]")
                names = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
                in_names_block = False
            else:
                names = []
                in_names_block = bool(val == "" or val == ":")
            continue
        if in_names_block:
            if s.startswith("-"):
                names.append(s.lstrip("-").strip().strip("'\""))
            else:
                in_names_block = False
        if s.startswith("nc:"):
            try:
                nc = int(s[len("nc:"):].strip())
            except ValueError:
                pass
    return nc, names


def _find_split_dirs(root):
    """在 Roboflow 导出的根目录（zip/文件夹）里找 train/valid/test 划分目录。"""
    found = []
    for name in ("train", "valid", "test", "val"):
        p = root / name
        if p.is_dir():
            found.append((name, p))
    return found


def _collect_split_files(split_dir):
    """递归收集 split 目录里图片和同名标签，返回 (stem -> path, stem -> label)。"""
    image_map = {}
    label_map = {}
    for p in sorted(split_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_SUFFIXES:
            image_map.setdefault(p.stem, p)
        elif p.suffix.lower() == ".txt" and p.stem.lower() != "classes":
            label_map.setdefault(p.stem, p)
    return image_map, label_map


def _unique_dest(dest_dir, name):
    """同名冲突时在扩展名前追加序号，避免覆盖已有数据。"""
    dest = dest_dir / name
    if not dest.exists():
        return dest, dest.stem
    stem, ext = dest.stem, dest.suffix
    for i in range(2, 1000):
        candidate = dest_dir / "{}_{}{}".format(stem, i, ext)
        if not candidate.exists():
            return candidate, "{}_{}".format(stem, i)
    raise RuntimeError("无法为 {} 找到不冲突的目标文件名".format(name))


def import_roboflow(source, dataset):
    """把 Roboflow 导出的 zip 或文件夹导入到我们标准的数据集目录结构。

    - train/valid/test 均被收集；valid 和 test 都并入 val（Ultralytics 用 val 做验证）
    - 类别信息从 Roboflow 的 data.yaml 解析；没有则按标签里最大 class_id 生成
    - 同名文件冲突时追加 _2、_3 序号，绝不覆盖
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError("Roboflow 数据不存在: {}".format(source))

    dataset_root = Path(dataset).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    if source.is_file() and source.suffix.lower() == ".zip":
        tmp_root = Path(dataset_root / "_roboflow_tmp")
        if tmp_root.exists():
            shutil.rmtree(str(tmp_root))
        tmp_root.mkdir(parents=True)
        try:
            import zipfile
            with zipfile.ZipFile(str(source)) as zf:
                zf.extractall(str(tmp_root))
            import_roboflow(tmp_root, dataset)
        finally:
            shutil.rmtree(str(tmp_root), ignore_errors=True)
        return
    if not source.is_dir():
        raise ValueError("--roboflow 必须是 .zip 文件或解压后的文件夹")

    split_dirs = _find_split_dirs(source)
    if not split_dirs:
        raise FileNotFoundError(
            "在 {} 里没找到 train/valid/test 目录，请确认是 Roboflow YOLO 导出格式".format(source))

    nc, names = None, None
    yaml_candidates = list(source.glob("data.yaml")) or list(source.rglob("data.yaml"))
    if yaml_candidates:
        nc, names = parse_data_yaml_text(
            yaml_candidates[0].read_text(encoding="utf-8", errors="ignore"))

    split_map = {"train": "train", "valid": "val", "val": "val", "test": "val"}
    dirs = dataset_dirs(dataset)
    for d in (dirs["train_images"], dirs["val_images"],
              dirs["train_labels"], dirs["val_labels"]):
        d.mkdir(parents=True, exist_ok=True)

    dest_dirs = {"train": dirs["train_images"], "val": dirs["val_images"]}
    dest_label_dirs = {"train": dirs["train_labels"], "val": dirs["val_labels"]}

    total_copied = 0
    skipped = []
    max_class_id = -1
    for split_name, split_dir in split_dirs:
        dest_split = split_map[split_name]
        image_map, label_map = _collect_split_files(split_dir)
        for stem, img_path in image_map.items():
            lbl_path = label_map.get(stem)
            if lbl_path is None:
                skipped.append("{}/{}".format(split_name, img_path.name))
                continue
            new_img, new_stem = _unique_dest(dest_dirs[dest_split], img_path.name)
            shutil.copy2(str(img_path), str(new_img))
            shutil.copy2(str(lbl_path), str(dest_label_dirs[dest_split] / (new_stem + ".txt")))
            try:
                for ln in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = ln.strip().split()
                    if parts:
                        max_class_id = max(max_class_id, int(parts[0]))
            except (ValueError, IndexError):
                pass
            total_copied += 1

    if names is None and max_class_id >= 0:
        n = (nc if nc is not None else max_class_id + 1)
        names = ["class_{}".format(i) for i in range(n)]
    if names is None:
        names = ["tube"]
    nc = len(names)

    root_str = str(dataset_root).replace("\\", "/")
    names_repr = "[" + ", ".join("'{}'".format(n) for n in names) + "]"
    yaml_text = ("path: {}\ntrain: images/train\nval: images/val\n"
                 "nc: {}\nnames: {}\n".format(root_str, nc, names_repr))
    (dataset_root / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("导入完成: 共复制 {} 张图片和标签".format(total_copied))
    if skipped:
        print("以下 {} 张图片没有对应标签，已跳过:".format(len(skipped)))
        for s in skipped[:10]:
            print("  -", s)
    print("data.yaml 已写入: nc={}, names={}".format(nc, names))
    print("\n开始数据集校验……")
    validate_dataset(str(dataset_root))


def validate_dataset(dataset):
    """校验数据集结构、图片/标签配对和标签格式，返回统计与问题列表。"""
    dirs = dataset_dirs(dataset)
    stats = {}
    issues = []
    nc = None
    yaml_path = Path(dataset).resolve() / "data.yaml"
    if yaml_path.is_file():
        _, names = parse_data_yaml_text(
            yaml_path.read_text(encoding="utf-8", errors="ignore"))
        if names:
            nc = len(names)
    for split in ("train", "val"):
        img_dir = dirs[split + "_images"]
        lbl_dir = dirs[split + "_labels"]
        if not img_dir.is_dir():
            issues.append("[{}] 图片目录不存在: {}".format(split, img_dir))
            continue
        if not lbl_dir.is_dir():
            issues.append("[{}] 标签目录不存在: {}".format(split, lbl_dir))
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        label_names = {p.stem for p in lbl_dir.glob("*.txt")}
        image_names = {p.stem for p in images}
        orphan_labels = sorted(label_names - image_names)
        if orphan_labels:
            issues.append("[{}] {} 个标签没有对应图片: {}".format(
                split, len(orphan_labels), orphan_labels[:5]))

        n_boxes = 0
        empty_labels = []
        missing_labels = []
        bad_labels = []
        for img_path in images:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.is_file():
                missing_labels.append(img_path.name)
                continue
            lines = [ln.strip() for ln in
                     lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                empty_labels.append(img_path.name)
                continue
            for ln_idx, ln in enumerate(lines, 1):
                parts = ln.split()
                if len(parts) != 5:
                    bad_labels.append("{}:{} 字段数={} (应为5)".format(
                        img_path.name, ln_idx, len(parts)))
                    continue
                try:
                    cls = int(parts[0])
                    vals = [float(x) for x in parts[1:]]
                except ValueError:
                    bad_labels.append("{}:{} 数值格式错误".format(img_path.name, ln_idx))
                    continue
                if cls < 0:
                    bad_labels.append("{}:{} 类别={} (不能为负)".format(
                        img_path.name, ln_idx, cls))
                elif nc is not None and cls >= nc:
                    bad_labels.append("{}:{} 类别={} 超出 data.yaml 的 nc={}".format(
                        img_path.name, ln_idx, cls, nc))
                if any(v <= 0.0 or v > 1.0 for v in vals):
                    bad_labels.append("{}:{} 坐标/尺寸超出 (0,1] 范围".format(
                        img_path.name, ln_idx))
                n_boxes += 1

        stats[split] = {
            "images": len(images),
            "boxes": n_boxes,
            "empty_labels": empty_labels,
            "missing_labels": missing_labels,
            "bad_labels": bad_labels,
        }
        if missing_labels:
            issues.append("[{}] {} 张图片没有标签文件".format(split, len(missing_labels)))
        if empty_labels:
            issues.append("[{}] {} 个标签为空（将被忽略）: {}".format(
                split, len(empty_labels), empty_labels[:5]))
        bad_count = len(bad_labels)
        if bad_count:
            issues.append("[{}] {} 条标签格式异常（前5条）: {}".format(
                split, bad_count, bad_labels[:5]))

    print("=" * 60)
    for split in ("train", "val"):
        if split in stats:
            s = stats[split]
            print("[{}] 图片 {} 张, 标注框 {} 个, 空标签 {}".format(
                split, s["images"], s["boxes"], len(s["empty_labels"])))
    print("=" * 60)
    if issues:
        print("发现 {} 个问题:".format(len(issues)))
        for item in issues:
            print("  - {}".format(item))
    else:
        print("数据集校验通过。")
    return stats, issues


def require_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("未安装 ultralytics，请先安装：")
        print("  {} -m pip install ultralytics".format(sys.executable))
        raise


def train_model(dataset, model_name, epochs, imgsz, batch, device, workers):
    YOLO = require_ultralytics()
    data_yaml = Path(dataset).resolve() / "data.yaml"
    if not data_yaml.is_file():
        write_data_yaml(dataset)
    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(Path(dataset).resolve() / "runs"),
        name="train",
        pretrained=True,
        patience=30,
        cache=False,
        # 数据增强：网络图片光照/角度/背景多样，适度增强有助泛化。
        degrees=5.0,
        translate=0.05,
        scale=0.25,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.0,   # 试管无色，不做色相偏移
        hsv_s=0.4,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.0,
        close_mosaic=10,
    )
    best = Path(dataset).resolve() / "runs" / "train" / "weights" / "best.pt"
    print("训练完成。最终权重: {}".format(best))


def tray_roi(width, height):
    """与旧算法保持一致的中央保守 ROI。"""
    x0 = int(round(0.05 * width))
    y0 = int(round(0.12 * height))
    x1 = int(round(0.95 * width))
    y1 = int(round(0.98 * height))
    return x0, y0, x1, y1


def imread_color(path):
    """读图并保证为 BGR 三通道，读取失败返回 None。"""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def run_detection(model, crop, x0, y0, x1, y1, imgsz, conf, iou, device, max_det):
    """对单帧（或 ROI 裁剪）推理，返回原图坐标系下已排序并编号的检测列表。

    坐标统一加上 ROI 偏移 (x0, y0)，与旧算法一致输出原图像素坐标。
    """
    results = model.predict(source=crop, imgsz=imgsz, conf=conf, iou=iou,
                            device=device, max_det=max_det, verbose=False)
    detections = []
    if results:
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            score = float(box.conf[0].cpu().item())
            bx0, by0, bx1, by1 = xyxy
            gx0, gy0 = bx0 + x0, by0 + y0
            gx1, gy1 = bx1 + x0, by1 + y0
            cx = (gx0 + gx1) / 2.0
            cy = (gy0 + gy1) / 2.0
            # 圆形试管取 min(宽,高)/2 作为半径，比均值更抗拉长框。
            radius = min(gx1 - gx0, gy1 - gy0) / 2.0
            detections.append({
                "x1": round(gx0, 2), "y1": round(gy0, 2),
                "x2": round(gx1, 2), "y2": round(gy1, 2),
                "x": round(cx, 2), "y": round(cy, 2),
                "radius": round(radius, 2),
                "confidence": round(score, 4),
            })
    # 与旧算法一致：按 y 再 x 排序，再编号，方便机械臂按格抓取（契约勿动）。
    detections.sort(key=lambda d: (d["y"], d["x"]))
    for idx, d in enumerate(detections, 1):
        d["id"] = idx
    return detections


def extract_center_points(detections):
    """提取各检测框中央点坐标（有几个框返回几个点）。

    返回轻量列表 [{"id": 1, "cx": ..., "cy": ...}, ...]，
    id 与排序沿用 predict 的 y→x 契约，可直接供机械臂/ROS 模块按格索引。
    """
    return [{"id": d["id"], "cx": d["x"], "cy": d["y"]}
            for d in detections]


def draw_detections(image, detections, roi=None, legacy_detections=None):
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()
    if roi is not None:
        x0, y0, x1, y1 = roi
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 180, 0), 2)
    for d in detections:
        a = (int(round(d["x1"])), int(round(d["y1"])))
        b = (int(round(d["x2"])), int(round(d["y2"])))
        c = (int(round(d["x"])), int(round(d["y"])))
        cv2.rectangle(canvas, a, b, (0, 255, 0), 2)
        cv2.drawMarker(canvas, c, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(canvas, "{} {:.2f}".format(d["id"], d["confidence"]),
                    (a[0], max(20, a[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if legacy_detections:
        for d in legacy_detections:
            c = (int(round(d["x"])), int(round(d["y"])))
            r = int(round(d["radius"]))
            cv2.circle(canvas, c, r, (255, 0, 255), 1)
    return canvas


def predict_model(input_path, output, weights, imgsz, conf, iou, device,
                  max_det, use_roi=False, compare=False):
    YOLO = require_ultralytics()
    model = YOLO(weights)
    legacy = None
    if compare:
        import importlib.util
        script = Path(__file__).resolve().parent / LEGACY_SCRIPT
        if not script.is_file():
            raise FileNotFoundError("找不到旧算法文件: {}".format(script))
        spec = importlib.util.spec_from_file_location("legacy_tube_detector", str(script))
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    all_summary = []
    for src in collect_images(input_path):
        image = imread_color(src)
        if image is None:
            print("跳过无法读取: {}".format(src))
            continue
        h, w = image.shape[:2]
        if use_roi:
            x0, y0, x1, y1 = tray_roi(w, h)
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        detections = run_detection(model, image[y0:y1, x0:x1], x0, y0, x1, y1,
                                   imgsz=imgsz, conf=conf, iou=iou,
                                   device=device, max_det=max_det)
        center_points = extract_center_points(detections)

        legacy_dets = None
        if legacy is not None:
            _, _, legacy_dets = legacy.detect_tubes(image)

        annotated = draw_detections(image, detections,
                                    (x0, y0, x1, y1) if use_roi else None,
                                    legacy_dets)
        cv2.imwrite(str(out / (src.stem + "_yolo标注.png")), annotated)
        with (out / (src.stem + "_yolo坐标.csv")).open(
                "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "x1", "y1", "x2", "y2", "x", "y",
                               "radius", "confidence"])
            writer.writeheader()
            writer.writerows(detections)
        record = {
            "image": str(src),
            "image_size": [w, h],
            "roi": [x0, y0, x1, y1],
            "count": len(detections),
            "detections": detections,
            "center_points": center_points,
        }
        if legacy_dets is not None:
            record["legacy_count"] = len(legacy_dets)
        (out / (src.stem + "_yolo坐标.json")).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        all_summary.append({"image": src.name, "count": len(detections)})
        msg = "{}: YOLO 检测到 {} 个试管".format(src.name, len(detections))
        if legacy_dets is not None:
            msg += "，旧算法检测到 {} 个".format(len(legacy_dets))
        print(msg)
    (out / "yolo汇总.json").write_text(
        json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n结果已保存到: {}".format(out.resolve()))
    return all_summary


def detect_centers(input_path, weights, imgsz=640, conf=0.25, iou=0.45,
                   device="cpu", max_det=128, use_roi=False):
    """进程内接口：推理图片/目录，只返回各检测框中央点坐标，供其他模块使用。

    单张图片 -> [{"id": 1, "cx": ..., "cy": ...}, ...]  （扁平中心点列表）
    目录     -> [{"image": 路径, "count": 数量, "points": [...]}, ...]  （逐图分组）

    - 坐标为原图像素坐标（ROI 内检测已自动加回偏移）；
    - id 按 y 再 x 连续编号（机械臂按格抓取契约，勿动）；
    - 改写 ROS 时，每帧以单图模式调用，返回值即可作为消息数据源。
    """
    YOLO = require_ultralytics()
    model = YOLO(weights)

    def _infer_one(src):
        image = imread_color(src)
        if image is None:
            return None
        h, w = image.shape[:2]
        if use_roi:
            x0, y0, x1, y1 = tray_roi(w, h)
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        detections = run_detection(model, image[y0:y1, x0:x1], x0, y0, x1, y1,
                                   imgsz=imgsz, conf=conf, iou=iou,
                                   device=device, max_det=max_det)
        return detections

    srcs = collect_images(input_path)
    if Path(input_path).is_file():
        detections = _infer_one(srcs[0])
        if detections is None:
            print("无法读取图片: {}".format(srcs[0]))
            return []
        return extract_center_points(detections)
    results = []
    for src in srcs:
        detections = _infer_one(src)
        if detections is None:
            print("跳过无法读取: {}".format(src))
            continue
        results.append({
            "image": str(src),
            "count": len(detections),
            "points": extract_center_points(detections),
        })
    return results


def export_model(weights, imgsz, device, half):
    YOLO = require_ultralytics()
    model = YOLO(weights)
    result = model.export(format="onnx", imgsz=imgsz, device=device, half=half,
                          simplify=False, dynamic=False)
    print("ONNX 导出完成: {}".format(result))


def main():
    parser = argparse.ArgumentParser(description="YOLO 试管检测（validate/train/predict/export）")
    parser.add_argument("--mode", choices=["import-roboflow", "validate", "train", "predict", "export"],
                        required=True)
    parser.add_argument("--roboflow", default="", help="Roboflow 导出的 .zip 或解压后的文件夹")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="YOLO 数据集目录")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="推理图片或目录")
    parser.add_argument("--output", default="检测结果yolo", help="推理输出目录")
    parser.add_argument("--weights", default="", help="best.pt 或 ONNX 权重")
    parser.add_argument("--model", default="yolo11n.pt", help="训练起始模型")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0, help="Windows 建议 0")
    parser.add_argument("--device", default="cpu", help="cpu、0 或 cuda:0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=128)
    parser.add_argument("--half", action="store_true", help="仅 CUDA/ONNX GPU 建议开启")
    parser.add_argument("--use-roi", action="store_true",
                        help="推理时裁剪中央 90%%x86%% ROI（与旧算法一致）")
    parser.add_argument("--compare", action="store_true",
                        help="同时运行旧腐蚀算法，在标注图上叠加对比")
    args = parser.parse_args()

    if args.mode == "import-roboflow":
        if not args.roboflow:
            parser.error("--mode import-roboflow 必须提供 --roboflow 路径")
        import_roboflow(args.roboflow, args.dataset)
    elif args.mode == "validate":
        validate_dataset(args.dataset)
    elif args.mode == "train":
        train_model(args.dataset, args.model, args.epochs, args.imgsz,
                    args.batch, args.device, args.workers)
    elif args.mode == "predict":
        if not args.weights:
            parser.error("--mode predict 必须提供 --weights best.pt")
        predict_model(args.input, args.output, args.weights, args.imgsz,
                      args.conf, args.iou, args.device, args.max_det,
                      args.use_roi, args.compare)
    elif args.mode == "export":
        if not args.weights:
            parser.error("--mode export 必须提供 --weights best.pt")
        export_model(args.weights, args.imgsz, args.device, args.half)


if __name__ == "__main__":
    main()
