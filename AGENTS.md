# AGENTS.md

## What this is

Single-script YOLO pipeline for detecting test tubes (试管) in tray photos, part of a bio-kit + robotic-arm project. Everything runs through `cs_试管_yolo检测.py` (no package, no tests, no linter config).

## Commands

All commands go through the one script; quote the filename in PowerShell:

```
python "cs_试管_yolo检测.py" --mode validate --dataset tube_dataset
python "cs_试管_yolo检测.py" --mode train --dataset tube_dataset
python "cs_试管_yolo检测.py" --mode predict --input <img-or-dir> --weights <best.pt>
python "cs_试管_yolo检测.py" --mode export --weights <best.pt>   # ONNX
python "cs_试管_yolo检测.py" --mode import-roboflow --roboflow <zip-or-dir> --dataset tube_dataset
```

- Dependencies: `ultralytics` and `opencv-python` (`cv2`); no requirements file. Missing `ultralytics` raises a friendly error.
- Mode order matters: import-roboflow → validate → train → predict/export. `predict` and `export` fail without `--weights`.
- Windows defaults baked in: `--workers 0`, `--device cpu`. Override with `--device 0` for CUDA.

## Gotchas

- `data.yaml` is auto-**rewritten with an absolute path** by `import-roboflow`/`validate`/`train`. A stale `path:` breaks training silently — check it if train/val fails to find images (existing `tube_dataset/data.yaml` currently points at a wrong parent dir).
- Training outputs go to `<dataset>/runs/train/weights/best.pt` (the script pins `project` inside the dataset dir, not Ultralytics' default `runs/`).
- Prediction writes `<stem>_yolo标注.png`, `_yolo坐标.csv` (UTF-8 BOM), `_yolo坐标.json` per image plus `yolo汇总.json` into `--output` (default `检测结果yolo/`).
- **Do not change the detection output contract**: coordinates are sorted by y then x, then assigned sequential `id` — the robotic arm relies on this ordering to grab by cell position.
- `--compare` dynamically imports `cs_腐蚀_3_最终版.py` (legacy erosion algorithm) from the script's directory; it is not in this folder, so `--compare` fails here.
- Default `--input` is hardcoded to `C:\Users\BlueLR\Pictures\OrangePi` (local machine path); always pass `--input` explicitly.

## Datasets (do not treat as duplicates)

- `tube_dataset/` — working dataset in the script's standard layout (`images/{train,val}`, `labels/{train,val}`).
- `tube_roboflow/` — already-imported Roboflow data with 2 classes `['0','1']` and a trained `runs/train/weights/best.pt`.
- `tube.v1i.yolov11/` — raw Roboflow export (train/valid/test splits); source for `import-roboflow`.
- `yolo11n.pt` — pretrained base model used as `--model` for training.

Label format: `class_id cx cy w h`, normalized 0–1. Only tubes (inserted wells) are labeled — empty wells and ribs are not.
