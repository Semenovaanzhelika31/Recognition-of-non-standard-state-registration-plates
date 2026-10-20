"""Процедурная генерация сцен с ГРЗ: фон, «автомобиль»/щит, перспектива, помехи.

Все изображения создаются самим скриптом (никаких внешних фотографий), поэтому
синтетическая часть датасета свободна от лицензионных ограничений.
Ограничение: сцены схематичны (кузов — набор примитивов), поэтому разрыв
доменов между синтетикой и реальными кадрами остаётся значительным.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

import gostplate as gp

PW, PH = 2000.0, 1500.0   # размер «плоскости объекта» в мм


# --------------------------------------------------------------------------- утилиты
def noise_tex(rng, h, w, octaves=4, base=4, persist=0.55):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = base * (2 ** o)
        a = rng.random((max(2, int(n * h / max(w, h))), max(2, int(n * w / max(w, h))))).astype(np.float32)
        out += amp * cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= persist
    out /= tot
    out -= out.min()
    out /= max(out.max(), 1e-6)
    return out


def rand_color(rng, lo=0, hi=255):
    return tuple(int(v) for v in rng.integers(lo, hi, 3))


CAR_COLORS = [(235, 235, 235), (20, 20, 20), (170, 170, 175), (90, 90, 95), (30, 30, 170), (160, 70, 30),
              (50, 110, 50), (60, 140, 190), (20, 30, 100), (200, 190, 150)]


def _rot(psi, th, ph):
    cz, sz = math.cos(psi), math.sin(psi)
    cy, sy = math.cos(th), math.sin(th)
    cx, sx = math.cos(ph), math.sin(ph)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx


def project(pts_mm, center_mm, yaw, pitch, roll, scale, f, cx, cy):
    """Проекция точек плоскости (x вправо, y вниз, мм) в пиксели."""
    R = _rot(roll, yaw, pitch)
    d = f / scale
    out = []
    for x, y in pts_mm:
        P = R @ np.array([x - center_mm[0], y - center_mm[1], 0.0])
        Z = P[2] + d
        out.append((f * P[0] / Z + cx, f * P[1] / Z + cy))
    return np.array(out, np.float32)


# --------------------------------------------------------------------------- фон
def make_background(rng, W, H):
    kind = rng.integers(4)
    sky = np.array(rand_color(rng, 120, 255), np.float32)
    gnd = np.array(rand_color(rng, 30, 150), np.float32)
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    hor = rng.uniform(0.25, 0.6)
    t = np.clip((y - hor) / 0.1, 0, 1)
    img = sky * (1 - t) + gnd * t
    img = np.broadcast_to(img, (H, W, 3)).copy()
    tex = noise_tex(rng, H, W, 5, rng.integers(3, 9))[..., None]
    img = img * (0.7 + 0.5 * tex)
    if kind == 0:  # здание: кирпич/плитка
        bh = int(rng.integers(14, 50)); bw = bh * 2
        col = np.array(rand_color(rng, 60, 200), np.float32)
        ys, xs = np.mgrid[0:H, 0:W]
        row = ys // bh
        off = (row % 2) * (bw // 2)
        mortar = (((xs + off) % bw) < 3) | ((ys % bh) < 3)
        brick = np.where(mortar[..., None], col * 1.3, col * (0.85 + 0.3 * rng.random((H, W, 1)).astype(np.float32)))
        m = (ys < H * rng.uniform(0.6, 0.9))[..., None]
        img = np.where(m, brick, img)
    elif kind == 1:  # забор/полосы
        k = int(rng.integers(6, 30))
        xs = (np.arange(W) // k) % 2
        stripes = np.where(xs[None, :, None] == 0, 0.6, 1.0).astype(np.float32)
        m = np.zeros((H, 1, 1), np.float32)
        m[: int(H * rng.uniform(0.4, 0.8))] = 1
        img = img * (1 - m + m * stripes)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # асфальт / текстура внизу
    return img


def add_text_clutter(img, rng, n):
    """Случайные надписи (вывески, наклейки) — жёсткие негативы для символьной детекции."""
    H, W = img.shape[:2]
    for _ in range(n):
        h = int(rng.integers(10, max(14, int(H * 0.12))))
        txt = "".join(gp.ALPHABET[rng.integers(len(gp.ALPHABET))] for _ in range(int(rng.integers(2, 10))))
        cw = int(h * rng.uniform(0.5, 0.8))
        w = cw * len(txt) + 12
        bgc = rand_color(rng, 120, 256) if rng.random() < 0.7 else rand_color(rng, 0, 120)
        fgc = rand_color(rng, 0, 90) if sum(bgc) > 380 else rand_color(rng, 160, 256)
        patch = np.empty((h + 12, w, 3), np.uint8)
        patch[:] = bgc
        fi = int(rng.integers(len(gp.FONT_FILES)))
        for i, ch in enumerate(txt):
            gp._paste_glyph(patch, ch, (6 + i * cw, 6, 6 + (i + 1) * cw - 2, 6 + h), fi, rng, 1.0, 0.0, fgc)
        ang = rng.normal(0, 6)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        al = cv2.warpAffine(np.full(patch.shape[:2], 255, np.uint8), M, (w, h + 12))
        pr = cv2.warpAffine(patch, M, (w, h + 12), borderMode=cv2.BORDER_REPLICATE)
        x0 = int(rng.integers(0, max(1, W - w))); y0 = int(rng.integers(0, max(1, H - h - 12)))
        roi = img[y0:y0 + h + 12, x0:x0 + w]
        a = (al[:roi.shape[0], :roi.shape[1]] / 255.0)[..., None]
        roi[:] = (roi * (1 - a) + pr[:roi.shape[0], :roi.shape[1]] * a).astype(np.uint8)


# --------------------------------------------------------------------------- объекты
def _fill(img, msk, shape_fn, color):
    shape_fn(img, color)
    shape_fn(msk, 255)


def draw_car_plane(rng, res, plate=None, plate_xy=None, front=None):
    """Кузов (вид спереди/сзади) на плоскости; возвращает (BGR, alpha)."""
    w, h = int(PW * res), int(PH * res)
    img = np.zeros((h, w, 3), np.uint8)
    msk = np.zeros((h, w), np.uint8)
    S = lambda v: int(round(v * res))
    body = CAR_COLORS[rng.integers(len(CAR_COLORS))]
    body = tuple(int(np.clip(c + rng.integers(-15, 16), 0, 255)) for c in body)
    dark = tuple(int(c * 0.35) for c in body)
    front = bool(rng.random() < 0.5) if front is None else front
    top = int(rng.integers(120, 420)); bot = int(rng.integers(1180, 1330))
    xl, xr = int(rng.integers(60, 200)), int(PW - rng.integers(60, 200))
    r = S(rng.integers(40, 160))

    def rrect(im, c, x0, y0, x1, y1, rad):
        cv2.rectangle(im, (S(x0) + rad, S(y0)), (S(x1) - rad, S(y1)), c, -1)
        cv2.rectangle(im, (S(x0), S(y0) + rad), (S(x1), S(y1) - rad), c, -1)
        for cx_, cy_ in [(S(x0) + rad, S(y0) + rad), (S(x1) - rad, S(y0) + rad), (S(x0) + rad, S(y1) - rad),
                         (S(x1) - rad, S(y1) - rad)]:
            cv2.circle(im, (cx_, cy_), rad, c, -1)

    # корпус
    rrect(img, body, xl, top + 250, xr, bot, r); rrect(msk, 255, xl, top + 250, xr, bot, r)
    # кабина/стекло
    gx0, gx1 = xl + int(rng.integers(120, 320)), xr - int(rng.integers(120, 320))
    gl = np.array([[S(gx0), S(top + 280)], [S(gx1), S(top + 280)], [S(gx1 - 120), S(top)], [S(gx0 + 120), S(top)]])
    glass = tuple(int(v) for v in rng.integers(15, 80, 3))
    cv2.fillPoly(img, [gl], body); cv2.fillPoly(msk, [gl], 255)
    inner = gl.copy(); inner[:, 0] = np.clip(inner[:, 0] + np.array([30, -30, -20, 20]) * res, 0, w)
    inner[:, 1] = np.clip(inner[:, 1] + np.array([-5, -5, 40, 40]) * res, 0, h)
    cv2.fillPoly(img, [inner], glass)
    # бампер
    by = int(rng.integers(930, 1030))
    cv2.rectangle(img, (S(xl + 20), S(by)), (S(xr - 20), S(bot)), tuple(int(c * rng.uniform(0.6, 0.95)) for c in body), -1)
    cv2.rectangle(img, (S(xl + 40), S(bot - 90)), (S(xr - 40), S(bot)), dark, -1)
    # фары / решётка
    lc = (230, 235, 240) if front else (30, 30, 190)
    for sx in (0, 1):
        x0 = xl + 40 if sx == 0 else xr - 40 - 340
        cv2.ellipse(img, (S(x0 + 170), S(by - 130)), (S(170), S(70)), 0, 0, 360, lc, -1)
    if front:
        gy0, gy1 = by - 120, by + 20
        cv2.rectangle(img, (S(520), S(gy0)), (S(1480), S(gy1)), (25, 25, 25), -1)
        for yy in range(int(gy0) + 10, int(gy1), 24):
            cv2.line(img, (S(530), S(yy)), (S(1470), S(yy)), (90, 90, 90), max(1, S(6)))
    # колёса
    for cx_ in (xl + 220, xr - 220):
        cv2.ellipse(img, (S(cx_), S(bot + 20)), (S(160), S(90)), 0, 0, 360, (15, 15, 15), -1)
        cv2.ellipse(msk, (S(cx_), S(bot + 20)), (S(160), S(90)), 0, 0, 360, 255, -1)
    # блики и градиент
    g = np.linspace(1.12, 0.78, h, dtype=np.float32)[:, None, None]
    img = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)
    # посадочное место и знак
    if plate is not None:
        pw_, ph_ = plate["img"].shape[1], plate["img"].shape[0]
        px, py = plate_xy
        m = int(14 * res)
        hold = tuple(int(v) for v in rng.integers(10, 60, 3))
        cv2.rectangle(img, (px - m, py - m), (px + pw_ + m, py + ph_ + m), hold, -1)
        cv2.rectangle(msk, (px - m, py - m), (px + pw_ + m, py + ph_ + m), 255, -1)
        if rng.random() < 0.5:   # рамка с надписью под знаком
            cv2.rectangle(img, (px - m, py + ph_), (px + pw_ + m, py + ph_ + int(40 * res)), (220, 220, 220), -1)
            cv2.rectangle(msk, (px - m, py + ph_), (px + pw_ + m, py + ph_ + int(40 * res)), 255, -1)
        paste_plate(img, plate, px, py)
    return img, msk


def paste_plate(img, plate, px, py):
    p, a = plate["img"], plate["alpha"][..., None]
    h, w = p.shape[:2]
    roi = img[py:py + h, px:px + w]
    if roi.shape[:2] != (h, w):
        return
    roi[:] = (roi * (1 - a) + p * a).astype(np.uint8)


def draw_board_plane(rng, res, plate, plate_xy):
    """Рекламный щит / витрина / экран с изображением знака (is_vehicle = 0)."""
    w, h = int(PW * res), int(PH * res)
    img = np.empty((h, w, 3), np.uint8)
    c1, c2 = np.array(rand_color(rng, 30, 250), np.float32), np.array(rand_color(rng, 30, 250), np.float32)
    t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    img[:] = np.clip(c1 * (1 - t) + c2 * t, 0, 255).astype(np.uint8)
    msk = np.full((h, w), 255, np.uint8)
    fr = max(2, int(30 * res))
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (25, 25, 25), fr)
    add_text_clutter(img, rng, int(rng.integers(1, 5)))
    paste_plate(img, plate, *plate_xy)
    return img, msk


# --------------------------------------------------------------------------- эффекты
def apply_dirt(plate_img, rng, amount):
    h, w = plate_img.shape[:2]
    n = noise_tex(rng, h, w, 5, 6)
    grad = np.linspace(0.4, 1.0, h, dtype=np.float32)[:, None]
    m = np.clip((n - (1 - amount)) * 3.0, 0, 1) * grad
    brown = np.array([40, 60, 85], np.float32)
    out = plate_img.astype(np.float32) * (1 - m[..., None] * 0.85) + brown * m[..., None] * 0.85
    return out.astype(np.uint8)


def motion_blur(img, rng):
    k = int(rng.integers(5, 26))
    ang = rng.uniform(0, 180) if rng.random() < 0.3 else rng.normal(0, 10)
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1)
    ker = cv2.warpAffine(ker, M, (k, k))
    ker /= max(ker.sum(), 1e-6)
    return cv2.filter2D(img, -1, ker)


def add_rain(img, rng):
    H, W = img.shape[:2]
    lay = np.zeros((H, W), np.float32)
    n = int(W * H / 2500)
    ang = rng.normal(0, 0.12)
    for _ in range(n):
        x, y = int(rng.integers(W)), int(rng.integers(H))
        L = int(rng.integers(8, 30))
        cv2.line(lay, (x, y), (int(x + L * math.sin(ang)), int(y + L * math.cos(ang))), 1.0, 1)
    lay = cv2.GaussianBlur(lay, (0, 0), 0.8)
    out = img.astype(np.float32) * 0.9 + lay[..., None] * rng.uniform(40, 110)
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.4))
    return np.clip(out, 0, 255).astype(np.uint8)


def add_snow(img, rng):
    H, W = img.shape[:2]
    lay = np.zeros((H, W), np.float32)
    for _ in range(int(W * H / 700)):
        cv2.circle(lay, (int(rng.integers(W)), int(rng.integers(H))), int(rng.integers(1, 3)), 1.0, -1)
    lay = cv2.GaussianBlur(lay, (0, 0), 0.7)
    out = img.astype(np.float32) * (1 - lay[..., None] * 0.9) + 255 * lay[..., None] * 0.9
    return np.clip(out, 0, 255).astype(np.uint8)


def add_glare(img, rng, cxy, size):
    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = cxy[0] + rng.normal(0, size * 0.3), cxy[1] + rng.normal(0, size * 0.2)
    sx, sy = size * rng.uniform(0.3, 0.9), size * rng.uniform(0.15, 0.5)
    g = np.exp(-(((xs - cx) / sx) ** 2 + ((ys - cy) / sy) ** 2))
    out = img.astype(np.float32) + g[..., None] * rng.uniform(120, 300)
    return np.clip(out, 0, 255).astype(np.uint8)


def night(img, rng, plate_center, plate_size):
    H, W = img.shape[:2]
    f = rng.uniform(0.12, 0.35)
    out = (img.astype(np.float32) / 255.0) ** rng.uniform(1.2, 1.8) * f * 255
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    sg = plate_size * rng.uniform(1.5, 3.5)
    spot = np.exp(-(((xs - plate_center[0]) ** 2 + (ys - plate_center[1]) ** 2) / (2 * sg ** 2)))
    out = out * (1 + spot[..., None] * rng.uniform(1.5, 5.0))
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- сцена
def make_scene(rng, W, H, spec=None):
    """Возвращает (img BGR, plates). spec: dict(types=[...]) для принудительного набора знаков.

    plates: [{type, text, quad(4x2, по часовой от левого верхнего), is_vehicle, conditions}]
    """
    bg = make_background(rng, W, H)
    add_text_clutter(bg, rng, int(rng.integers(0, 6)))
    r = rng.random()
    n_obj = 0 if r < 0.08 else (1 if r < 0.88 else 2)
    plates, conds = [], set()
    img = bg
    night_flag = rng.random() < 0.22
    objs = []
    for oi in range(n_obj):
        # --- тип знака
        u = rng.random()
        if spec and spec.get("types"):
            ptype = spec["types"][oi % len(spec["types"])]
        elif u < 0.32:
            ptype = "type1"
        elif u < 0.56:
            ptype = "type1a"
        elif u < 0.80:
            ptype = "type1b"
        else:
            ptype = "other"
        is_vehicle = 1 if rng.random() < 0.88 else 0
        # --- геометрия камеры
        wp_px = float(np.exp(rng.uniform(np.log(42), np.log(min(560, W * 0.6))))) * (0.7 if oi else 1.0)
        base_w = 520.0 if ptype != "type1a" else 290.0
        scale = wp_px / 520.0                         # пикселей на мм (для широкого знака)
        if ptype == "type1a":
            scale = wp_px / 290.0 * 0.62
        res = float(np.clip(scale * 1.6, 0.25, 2.0))
        plate_res = res
        if ptype == "other":
            ok = gp.OTHER_KINDS[rng.integers(len(gp.OTHER_KINDS))]
            plate = gp.render_other(ok, plate_res, rng)
        else:
            text = gp.random_text(rng)
            plate = gp.render_plate(ptype, text, plate_res, rng)
        ph_mm, pw_mm = plate["h_mm"], plate["w_mm"]
        px_mm = (PW - pw_mm) / 2 + rng.normal(0, 40)
        py_mm = float(rng.uniform(960, 1150)) if is_vehicle else float(rng.uniform(150, PH - ph_mm - 150))
        if not is_vehicle:
            px_mm = float(rng.uniform(150, PW - pw_mm - 150))
        px_mm = float(np.clip(px_mm, 80, PW - pw_mm - 80))
        pxy = (int(px_mm * res), int(py_mm * res))
        if rng.random() < 0.55:
            plate["img"] = apply_dirt(plate["img"], rng, rng.uniform(0.15, 0.45)); conds.add("dirt")
        if is_vehicle:
            tex, am = draw_car_plane(rng, res, plate, pxy)
        else:
            tex, am = draw_board_plane(rng, res, plate, pxy)
        # --- гомография плоскости в кадр
        yaw = float(np.clip(rng.normal(0, 22), -58, 58)) * math.pi / 180
        pitch = float(np.clip(rng.normal(0, 6), -20, 20)) * math.pi / 180
        roll = float(np.clip(rng.normal(0, 3), -12, 12)) * math.pi / 180
        if abs(yaw) > 0.43 or abs(pitch) > 0.26:
            conds.add("angle")
        f = W * rng.uniform(0.8, 1.6)
        pc = (px_mm + pw_mm / 2, py_mm + ph_mm / 2)
        scx = float(rng.uniform(0.12, 0.88)) * W
        scy = float(rng.uniform(0.15, 0.9)) * H
        corners_plane = [(0, 0), (PW, 0), (PW, PH), (0, PH)]
        dst = project(corners_plane, pc, yaw, pitch, roll, scale, f, scx, scy)
        pq = project([(px_mm, py_mm), (px_mm + pw_mm, py_mm), (px_mm + pw_mm, py_mm + ph_mm), (px_mm, py_mm + ph_mm)],
                     pc, yaw, pitch, roll, scale, f, scx, scy)
        m = max(2.0, 0.04 * wp_px)
        if pq[:, 0].min() < m or pq[:, 1].min() < m or pq[:, 0].max() > W - m or pq[:, 1].max() > H - m:
            continue
        if not np.isfinite(dst).all() or np.abs(dst).max() > 20 * max(W, H):
            continue
        src = np.array([[0, 0], [tex.shape[1], 0], [tex.shape[1], tex.shape[0]], [0, tex.shape[0]]], np.float32)
        Hm = cv2.getPerspectiveTransform(src, dst)
        factor = scale / res
        if factor < 0.7:
            sg = 0.45 / factor
            tex = cv2.GaussianBlur(tex, (0, 0), sg)
        warped = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        wm = cv2.warpPerspective(am, Hm, (W, H), flags=cv2.INTER_LINEAR)
        wm = cv2.GaussianBlur(wm, (0, 0), 0.6).astype(np.float32)[..., None] / 255.0
        img = (img * (1 - wm) + warped * wm).astype(np.uint8)
        # --- метка
        sp = pq.reshape(-1, 2)
        plate_text = plate["text"] if ptype != "other" else ""
        plates.append(dict(type=ptype, text=plate_text, quad=sp, is_vehicle=is_vehicle, scale=wp_px))
        objs.append((np.mean(sp, 0), wp_px))
    # --- эффекты всего кадра
    if n_obj and objs:
        cen, wsz = objs[0]
    else:
        cen, wsz = (W / 2, H / 2), 80
    if night_flag:
        img = night(img, rng, cen, wsz); conds.add("night")
    else:
        conds.add("day")
        gain = rng.uniform(0.75, 1.15)
        img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if rng.random() < 0.14:
        img = add_rain(img, rng); conds.add("rain")
    if rng.random() < 0.08:
        img = add_snow(img, rng); conds.add("snow")
    if rng.random() < 0.14:
        img = add_glare(img, rng, cen, wsz * 0.8); conds.add("glare")
    if rng.random() < 0.2:
        img = motion_blur(img, rng); conds.add("motion_blur")
    elif rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.8))
    if rng.random() < 0.25:   # потеря разрешения
        k = rng.uniform(0.4, 0.8)
        img = cv2.resize(cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), (W, H))
    sig = rng.uniform(0, 4) if not night_flag else rng.uniform(3, 11)
    if sig > 0.5:
        img = np.clip(img.astype(np.float32) + rng.normal(0, sig, img.shape), 0, 255).astype(np.uint8)
    cond_list = sorted(conds)
    for p in plates:
        p["conditions"] = cond_list
    return img, plates
