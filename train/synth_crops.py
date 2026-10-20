"""Синтетические выпрямленные кропы знаков для обучения распознавателя.

Имитируют тракт «кадр -> детектор -> четырёхугольник -> гомография»: знак рисуется,
проецируется в кадр с перспективой, портится помехами, затем выпрямляется по
четырёхугольнику, сдвинутому случайной ошибкой (как ошибается детектор).
"""
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anpr import gost  # noqa: E402  (добавляет generator в sys.path)
import gostplate as gp  # noqa: E402
import scenes as sc  # noqa: E402


def night_mild(img, rng, plate_center, plate_size):
    """Ночь для обучения распознавателя: мягче, чем в сценах (без полного пересвета)."""
    H, W = img.shape[:2]
    f = rng.uniform(0.2, 0.5)
    out = (img.astype(np.float32) / 255.0) ** rng.uniform(1.1, 1.5) * f * 255
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    sg = plate_size * rng.uniform(1.5, 3.5)
    spot = np.exp(-(((xs - plate_center[0]) ** 2 + (ys - plate_center[1]) ** 2) / (2 * sg ** 2)))
    out = out * (1 + spot[..., None] * rng.uniform(0.8, 2.5))
    return np.clip(out, 0, 255).astype(np.uint8)


def jitter_quad(quad, rng, mult=1.0):
    q = quad.astype(np.float64).copy()
    tl, tr, br, bl = q
    ex = (tr - tl + br - bl) / 2
    ey = (bl - tl + br - tr) / 2
    w, h = np.linalg.norm(ex), np.linalg.norm(ey)
    ux, uy = ex / max(w, 1e-6), ey / max(h, 1e-6)
    c = q.mean(0)
    # глобальные ошибки
    d = (rng.normal(0, 0.035 * w * mult) * ux + rng.normal(0, 0.07 * h * mult) * uy)
    sx, sy = rng.normal(1, 0.04 * mult), rng.normal(1, 0.05 * mult)
    a = math.radians(rng.normal(0, 1.5 * mult))
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    out = []
    for p in q:
        v = p - c
        vx, vy = v @ ux, v @ uy
        v = (vx * sx) * ux + (vy * sy) * uy
        v = R @ v
        out.append(c + v + d + rng.normal(0, 1, 2) * (0.01 * w * mult * ux + 0.03 * h * mult * uy))
    return np.array(out, np.float32)


def rectify_gray(img, quad, ptype):
    cw, ch = gost.canon_size(ptype)
    wq = np.linalg.norm(quad[1] - quad[0])
    factor = cw / max(wq, 1.0)
    if factor < 0.7:
        img = cv2.GaussianBlur(img, (0, 0), 0.45 / factor)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), gost.canon_corners(ptype))
    return cv2.warpPerspective(img, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def sample_crop(rng, ptype=None, jit=None):
    """Возвращает dict(gray, bgr, ptype, text, region_len)."""
    if ptype is None:
        ptype = ["type1", "type1a", "type1b"][rng.choice(3, p=[0.36, 0.34, 0.30])]
    text = gp.random_text(rng)
    s_img = float(np.exp(rng.uniform(np.log(0.075), np.log(1.0))))
    if ptype == "type1a":
        s_img *= 0.9
    res = float(np.clip(s_img * 1.6, 0.25, 1.5))
    plate = gp.render_plate(ptype, text, res, rng)
    if rng.random() < 0.5:
        plate["img"] = sc.apply_dirt(plate["img"], rng, rng.uniform(0.12, 0.5))
    wmm, hmm = plate["w_mm"], plate["h_mm"]
    m = 30
    tw, th = int((wmm + 2 * m) * res), int((hmm + 2 * m) * res)
    body = sc.CAR_COLORS[rng.integers(len(sc.CAR_COLORS))]
    tex = np.empty((th, tw, 3), np.uint8)
    tex[:] = body
    hold = tuple(int(v) for v in rng.integers(10, 70, 3))
    mh = int(14 * res)
    px, py = int(m * res), int(m * res)
    cv2.rectangle(tex, (px - mh, py - mh), (px + plate["img"].shape[1] + mh, py + plate["img"].shape[0] + mh), hold, -1)
    sc.paste_plate(tex, plate, px, py)
    # перспектива
    yaw = float(np.clip(rng.normal(0, 24), -62, 62)) * math.pi / 180
    pitch = float(np.clip(rng.normal(0, 7), -22, 22)) * math.pi / 180
    roll = float(np.clip(rng.normal(0, 4), -14, 14)) * math.pi / 180
    f = 1000.0
    corners = [(0, 0), (wmm + 2 * m, 0), (wmm + 2 * m, hmm + 2 * m), (0, hmm + 2 * m)]
    cen = ((wmm + 2 * m) / 2, (hmm + 2 * m) / 2)
    dst = sc.project(corners, cen, yaw, pitch, roll, s_img, f, 0.0, 0.0)
    pq = sc.project([(m, m), (m + wmm, m), (m + wmm, m + hmm), (m, m + hmm)], cen, yaw, pitch, roll, s_img, f, 0, 0)
    x0, y0 = dst.min(0); x1, y1 = dst.max(0)
    mg = 0.35 * max(x1 - x0, 20)
    W, H = int(x1 - x0 + 2 * mg), int(y1 - y0 + 2 * mg)
    H = max(H, 12); W = max(W, 12)
    off = np.array([mg - x0, mg - y0], np.float32)
    dst, pq = dst + off, pq + off
    # фон-окружение
    bgc = np.array(sc.rand_color(rng, 20, 235), np.float32)
    bg = (bgc * (0.65 + 0.7 * sc.noise_tex(rng, H, W, 4, 3)[..., None])).astype(np.uint8)
    src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    factor = s_img / res
    if factor < 0.7:
        tex = cv2.GaussianBlur(tex, (0, 0), 0.45 / factor)
    img = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wm = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), Hm, (W, H)).astype(np.float32)[..., None] / 255
    img = (bg * (1 - wm) + img * wm).astype(np.uint8)
    # помехи
    wpx = float(np.linalg.norm(pq[1] - pq[0]))
    cc = pq.mean(0)
    if rng.random() < 0.22:
        img = night_mild(img, rng, cc, wpx)
    else:
        img = np.clip(img.astype(np.float32) * rng.uniform(0.7, 1.2), 0, 255).astype(np.uint8)
    if rng.random() < 0.12:
        img = sc.add_rain(img, rng)
    if rng.random() < 0.07:
        img = sc.add_snow(img, rng)
    if rng.random() < 0.14:
        img = sc.add_glare(img, rng, cc, wpx * 0.7)
    if rng.random() < 0.2:
        img = sc.motion_blur(img, rng)
    elif rng.random() < 0.25:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.6))
    sig = rng.uniform(0, 4)
    if sig > 0.5:
        img = np.clip(img.astype(np.float32) + rng.normal(0, sig, img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.6:   # JPEG
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 95))])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    mult = jit if jit is not None else float(rng.uniform(0.2, 1.6))
    quad_est = jitter_quad(pq, rng, mult)
    gray_src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    C = rectify_gray(gray_src, quad_est, ptype)
    return dict(gray=C, ptype=ptype, text=text, region_len=len(text) - 6, wpx=wpx)
