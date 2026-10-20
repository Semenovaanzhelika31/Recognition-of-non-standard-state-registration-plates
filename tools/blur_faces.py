#!/usr/bin/env python3
"""Автоматическое размытие лиц (необязательный помощник; результат ОБЯЗАТЕЛЬНО просмотреть глазами).

Требует модель YuNet (MIT, https://github.com/opencv/opencv_zoo, face_detection_yunet_2023mar.onnx),
которую нужно скачать самостоятельно: в окружении, где писался проект, загрузка была закрыта,
поэтому скрипт там НЕ запускался. Ручное размытие есть в tools/annotate.py (клавиша b).

    python tools/blur_faces.py --src in_dir --dst out_dir --model face_detection_yunet_2023mar.onnx
"""
import argparse, os
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
ap.add_argument("--model", required=True); ap.add_argument("--score", type=float, default=0.6)
a = ap.parse_args()
os.makedirs(a.dst, exist_ok=True)
for fn in sorted(os.listdir(a.src)):
    if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = cv2.imread(os.path.join(a.src, fn))
    if img is None:
        continue
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(a.model, "", (w, h), a.score)
    _, faces = det.detect(img)
    n = 0
    for f in (faces if faces is not None else []):
        x, y, bw, bh = [int(v) for v in f[:4]]
        pad = int(0.25 * max(bw, bh))
        x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + bw + pad, w), min(y + bh + pad, h)
        roi = img[y0:y1, x0:x1]
        k = max(5, (min(roi.shape[:2]) // 2) | 1)
        img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0); n += 1
    cv2.imwrite(os.path.join(a.dst, fn), img)
    print(fn, "лиц размыто:", n)
