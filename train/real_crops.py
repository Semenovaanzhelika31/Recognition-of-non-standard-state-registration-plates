"""Реальные вырезки номеров (внешний набор Kaggle «Russian License Plate Characters - 23 Classes»,
CC BY 4.0, автор dimahkiin; разметка символов сделана автором моделью — это внешние данные,
НЕ ручная разметка и НЕ часть собственного реального датасета).

Из рамок символов собирается текст номера (сортировка по x), знак выпрямляется в канонический
вид аффинным преобразованием по рамкам, затем идёт обычное обучение распознавателя.
"""
import glob
import hashlib
import os
import re

import cv2
import numpy as np

from anpr import gost

NAMES = "0123456789ABCEHKMOPTXYD"            # порядок классов набора
LET = "ABEKMHOPCTYX"
RX = re.compile(r"^[%s][0-9]{3}[%s]{2}[0-9]{2,3}$" % (LET, LET))


def load_labels(path):
    rows = [l.split() for l in open(path, encoding="utf-8") if l.strip()]
    boxes = []
    for r in rows:
        c = int(r[0])
        if c >= len(NAMES):
            return None
        boxes.append((float(r[1]), float(r[2]), float(r[3]), float(r[4]), NAMES[c]))
    boxes.sort(key=lambda b: b[0])
    return boxes


def list_samples(images_dir, labels_dir):
    """[(image_path, text, boxes_px)] только для номеров, проходящих маску ГРЗ."""
    out = []
    for lp in sorted(glob.glob(os.path.join(labels_dir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(lp))[0]
        ip = None
        for ext in (".jpeg", ".jpg", ".png"):
            if os.path.exists(os.path.join(images_dir, stem + ext)):
                ip = os.path.join(images_dir, stem + ext)
                break
        if ip is None:
            continue
        b = load_labels(lp)
        if not b:
            continue
        text = "".join(x[4] for x in b)
        if RX.match(text):
            out.append((ip, text, b))
    return out


def split_by_text(samples, test_frac=0.2, salt="plates-real-v1"):
    """Разбиение по тексту номера (один номер — одна часть), чтобы кадры одной машины не протекали."""
    tr, te = [], []
    for s in samples:
        h = int(hashlib.sha1((salt + s[1]).encode()).hexdigest(), 16) % 1000
        (te if h < test_frac * 1000 else tr).append(s)
    return tr, te


def fit_canon(img, text, boxes, jitter=None, rng=None):
    """Выпрямляет вырезку в канонический вид type1 по рамкам символов. Возвращает gray или None."""
    h, w = img.shape[:2]
    rl = len(text) - 6
    slots = gost.canon_slots("type1", rl)
    main = [i for i, (k, _) in enumerate(slots) if k != "region"] if isinstance(slots[0][0], str) else list(range(6))
    cx = np.array([b[0] * w for b in boxes]); cy = np.array([b[1] * h for b in boxes])
    bh = np.array([b[3] * h for b in boxes]); bw = np.array([b[2] * w for b in boxes])
    sx_c = np.array([(s[1][0] + s[1][2]) / 2 for s in slots]); sy_c = np.array([(s[1][1] + s[1][3]) / 2 for s in slots])
    sh_c = np.array([s[1][3] - s[1][1] for s in slots])
    # x: центр = a*x + b (МНК); y: масштаб по высоте символов, сдвиг по центрам
    A = np.vstack([cx, np.ones_like(cx)]).T
    (ax, bx), *_ = np.linalg.lstsq(A, sx_c, rcond=None)
    # высоты: берём первые 6 символов (серия), регион мельче
    ay = float(np.median(sh_c[:6] / np.maximum(bh[:6], 1.0)))
    by = float(np.mean(sy_c[:6] - ay * cy[:6]))
    if not (np.isfinite(ax) and ax > 0 and ay > 0):
        return None
    M = np.array([[ax, 0, bx], [0, ay, by]], np.float64)
    if jitter is not None:
        rng = rng or np.random.default_rng(0)
        cw, ch = gost.canon_size("type1")
        J = np.eye(3)
        ang = np.deg2rad(rng.normal(0, jitter * 1.5))
        sc = np.exp(rng.normal(0, 0.04 * jitter))
        J[:2, :2] = sc * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        J[0, 2] = rng.normal(0, 0.03 * cw * jitter)
        J[1, 2] = rng.normal(0, 0.05 * ch * jitter)
        c = np.array([cw / 2, ch / 2])
        J[:2, 2] += c - J[:2, :2] @ c
        M = (J @ np.vstack([M, [0, 0, 1]]))[:2]
    cw, ch = gost.canon_size("type1")
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.warpAffine(g, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
