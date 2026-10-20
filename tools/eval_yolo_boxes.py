#!/usr/bin/env python3
"""Проверка детекции и типа знака на внешнем наборе с рамками в формате YOLO
(например, Kaggle «Russian license plates: 5-class detection»).

Номера в таком наборе не размечены, поэтому проверяется ТОЛЬКО: нашёл ли пайплайн знак
(рамка совпала по IoU) и совпал ли тип (taxi -> type1b, civilian -> type1/type1a).
Чужие снимки используются только для измерения, НЕ для обучения и НЕ для сдаваемого датасета.

    python tools/eval_yolo_boxes.py --root D:\\data\\plates5 --split test
    # ожидается root/images/<split>/*.jpg и root/labels/<split>/*.txt
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anpr.pipeline import PlateReader  # noqa: E402

NAMES = {0: "civilian", 1: "taxi", 2: "military", 3: "police", 4: "diplomatic"}
TARGET = {0, 1}                      # military/police/diplomatic — вне целевых типов, пайплайн их не выдаёт


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_labels(path, W, H):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
        out.append((c, [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--iou", type=float, default=0.4)
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    a = ap.parse_args()
    files = sorted(sum((glob.glob(os.path.join(a.root, "images", a.split, e)) for e in ("*.jpg", "*.jpeg", "*.png")), []))
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit("нет изображений в %s" % os.path.join(a.root, "images", a.split))
    reader = PlateReader(a.models)
    n_gt = {c: 0 for c in NAMES}
    hit = {c: 0 for c in NAMES}
    type_ok = {0: 0, 1: 0}
    pred_total = pred_matched = 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        H, W = img.shape[:2]
        lab = read_labels(os.path.join(a.root, "labels", a.split, os.path.splitext(os.path.basename(f))[0] + ".txt"), W, H)
        preds = reader.read(img)
        pb = []
        for p in preds:
            q = np.asarray(p["quad"])
            pb.append(([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()], p["plate_type"]))
        for c, g in lab:
            n_gt[c] += 1
            best = max(((box_iou(g, b), t) for b, t in pb), default=(0.0, None), key=lambda x: x[0])
            if best[0] >= a.iou:
                hit[c] += 1
                if c == 1 and best[1] == "type1b":
                    type_ok[1] += 1
                if c == 0 and best[1] in ("type1", "type1a"):
                    type_ok[0] += 1
        for b, _ in pb:
            pred_total += 1
            if any(box_iou(g, b) >= a.iou for _, g in lab):
                pred_matched += 1
    print("изображений: %d" % len(files))
    for c in NAMES:
        if n_gt[c]:
            tag = "" if c in TARGET else "  (вне целевых типов, пайплайн их не выдаёт)"
            print("%-10s знаков %4d, найдено %4d (%.2f)%s" % (NAMES[c], n_gt[c], hit[c], hit[c] / n_gt[c], tag))
    if hit[1]:
        print("taxi найдены как type1b: %d из %d" % (type_ok[1], hit[1]))
    if hit[0]:
        print("civilian найдены как type1/type1a: %d из %d" % (type_ok[0], hit[0]))
    print("предсказаний %d, из них совпало с размеченным знаком %d (%.2f)" %
          (pred_total, pred_matched, pred_matched / max(pred_total, 1)))


if __name__ == "__main__":
    main()
