#!/usr/bin/env python3
"""Контрольный набор «реальный знак на синтетическом фоне» для проверки детектора на реальной внешности знаков.

Берёт отложенные (по тексту номера) реальные вырезки внешнего набора Kaggle
«Russian License Plate Characters — 23 Classes» (CC BY 4.0, dimahkiin; разметка автора автоматическая),
вставляет каждую в синтетический кадр в случайном масштабе (0.35–1.6 от размера вырезки), опционально на
«кузов», и сохраняет кадры и meta.csv в том же формате, что и синтетический датасет.
Эти кадры предназначены ТОЛЬКО для измерения: в датасет и в обучение не входят.

    python tools/make_real_bench.py --images IMG/val --labels LAB/val --out BENCH --n 600
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anpr import gost  # noqa: E402
from train import real_crops as rc  # noqa: E402
import scenes as sc  # noqa: E402  (путь к generator добавляет anpr.gost)

FIELDS = ["image", "plate_num", "plate_type", "bbox", "quad", "is_vehicle", "is_synthetic", "source", "license",
          "conditions"]


def oracle_quad(img_shape, text, boxes):
    """Четырёхугольник знака в координатах вырезки по положению символов (обратная аффинная к fit_canon)."""
    h, w = img_shape[:2]
    rl = len(text) - 6
    slots = gost.canon_slots("type1", rl)
    cx = np.array([b[0] * w for b in boxes]); cy = np.array([b[1] * h for b in boxes])
    bh = np.array([b[3] * h for b in boxes])
    sx = np.array([(s[1][0] + s[1][2]) / 2 for s in slots]); sy = np.array([(s[1][1] + s[1][3]) / 2 for s in slots])
    sh = np.array([s[1][3] - s[1][1] for s in slots])
    A = np.vstack([cx, np.ones_like(cx)]).T
    (ax, bx), *_ = np.linalg.lstsq(A, sx, rcond=None)
    ay = float(np.median(sh[:6] / np.maximum(bh[:6], 1)))
    by = float(np.mean(sy[:6] - ay * cy[:6]))
    M = np.array([[ax, 0, bx], [0, ay, by], [0, 0, 1]])
    cw, ch = gost.canon_size("type1")
    corners = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float64).T
    return (np.linalg.inv(M) @ np.vstack([corners, np.ones(4)]))[:2].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--smin", type=float, default=0.35)
    ap.add_argument("--smax", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()
    _, test = rc.split_by_text(rc.list_samples(a.images, a.labels))
    os.makedirs(os.path.join(a.out, "images", "composite"), exist_ok=True)
    rows = []
    for j, (ip, text, boxes) in enumerate(test[:a.n]):
        rng = np.random.default_rng(np.random.SeedSequence([a.seed, j]))
        crop = cv2.imread(ip)
        q_crop = oracle_quad(crop.shape, text, boxes)
        W, H = [(640, 480), (960, 540), (1280, 720)][rng.choice(3, p=[.4, .3, .3])]
        s = float(np.exp(rng.uniform(np.log(a.smin), np.log(a.smax))))
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        cr = cv2.resize(crop, None, fx=s, fy=s, interpolation=interp)
        ch, cw = cr.shape[:2]
        if cw > 0.8 * W or ch > 0.8 * H:
            continue
        bg = sc.make_background(rng, W, H)
        sc.add_text_clutter(bg, rng, int(rng.integers(0, 5)))
        x0 = int(rng.integers(int(0.05 * W), W - cw - int(0.05 * W)))
        y0 = int(rng.integers(int(0.1 * H), H - ch - int(0.1 * H)))
        if rng.random() < 0.6:   # «кузов» вокруг знака
            pad = int(rng.uniform(0.15, 0.6) * ch)
            col = sc.CAR_COLORS[rng.integers(len(sc.CAR_COLORS))]
            cv2.rectangle(bg, (max(x0 - 3 * pad, 0), max(y0 - 2 * pad, 0)), (min(x0 + cw + 3 * pad, W - 1), min(y0 + ch + pad, H - 1)), col, -1)
        bg[y0:y0 + ch, x0:x0 + cw] = cr
        if rng.random() < 0.3:
            bg = np.clip(bg.astype(np.float32) + rng.normal(0, rng.uniform(1, 4), bg.shape), 0, 255).astype(np.uint8)
        q = (q_crop * s + [x0, y0]).astype(np.float32)
        name = "real_%05d" % j
        cv2.imwrite(os.path.join(a.out, "images", "composite", name + ".jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 96))])
        xs, ys = q[:, 0], q[:, 1]
        rows.append(dict(image=f"images/composite/{name}.jpg", plate_num=text, plate_type="type1",
                         bbox="%d,%d,%d,%d" % (xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()),
                         quad=",".join("%d" % round(v) for v in q.reshape(-1)), is_vehicle=1, is_synthetic=0,
                         source="kaggle dimahkiin 23-classes (CC BY 4.0) on synthetic background", license="CC BY 4.0",
                         conditions="day"))
    with open(os.path.join(a.out, "meta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter=";")
        w.writeheader(); w.writerows(rows)
    print("кадров:", len(rows))


if __name__ == "__main__":
    main()
