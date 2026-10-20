"""Рендер государственных регистрационных знаков РФ типов 1, 1А, 1Б.

Геометрия задана в миллиметрах и приближает ГОСТ Р 50577-2018 (габариты
знаков 520x112 / 290x170 мм и состав «буква, три цифры, две буквы, код региона»
взяты из ТЗ; размеры символов, интервалы и положение блока региона оценены
по общеизвестному виду знаков и могут отличаться от стандарта на единицы мм).
Шрифты — свободные (см. fonts/*-LICENSE.txt); форма глифов не совпадает с
эталонным шрифтом ГОСТ, это учитывается аугментацией (смена шрифта, ширины и
толщины штриха).

Модуль не зависит от остальной части проекта и используется и генератором
датасета, и обучением распознавателя, и самим распознавателем (раскладка слотов).
"""
from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
try:  # Pillow нужен только для рендера знаков (генератор/обучение), не для run.py
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = ImageDraw = ImageFont = None

LETTERS = "ABEKMHOPCTYX"          # допустимые буквы (латиница, ВЕРХНИЙ РЕГИСТР)
DIGITS = "0123456789"
ALPHABET = LETTERS + DIGITS

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_FILES = ["DejaVuSansCondensed-Bold.ttf", "LiberationSans-Bold.ttf",
              "Carlito-Bold.ttf", "Inter-Bold.otf"]

# коды регионов: двузначные 01..99 и распространённые трёхзначные (начинаются с 1, 2, 7)
REGIONS2 = [f"{i:02d}" for i in range(1, 100)]
REGIONS3 = ["102", "113", "116", "121", "123", "124", "125", "134", "136", "138", "142",
            "150", "152", "154", "159", "161", "163", "164", "173", "174", "177", "178",
            "186", "190", "196", "197", "198", "199", "277", "299", "702", "716", "750",
            "761", "763", "774", "777", "790", "797", "799"]

SIZE_MM = {"type1": (520, 112), "type1b": (520, 112), "type1a": (290, 170)}

BG_WHITE = (244, 244, 240)      # RGB
BG_YELLOW = (250, 196, 20)


# --------------------------------------------------------------------------- layout
def layout(ptype: str, region_len: int = 2):
    """Слоты символов знака: список (kind, (x0, y0, x1, y1) в мм).

    kind: 'L' — буква серии, 'D' — цифра номера, 'R' — цифра региона.
    Порядок слотов совпадает с порядком символов в номере A123BC77.
    """
    if ptype in ("type1", "type1b"):
        xs = [25, 85, 145, 205, 265, 325]
        kinds = "LDDDLL"
        slots = [(k, (x, 16, x + 50, 96)) for k, x in zip(kinds, xs)]
        if region_len == 2:
            rx = [400, 450]; rw = 42
        else:
            rx = [393, 433, 473]; rw = 34
        slots += [("R", (x, 14, x + rw, 74)) for x in rx]
    elif ptype == "type1a":
        top = [30, 78, 126, 174]
        slots = [(k, (x, 14, x + 40, 76)) for k, x in zip("LDDD", top)]
        slots += [("L", (x, 88, x + 40, 150)) for x in (30, 78)]
        if region_len == 2:
            rx = [140, 182]; rw = 36
        else:
            rx = [136, 168, 200]; rw = 28
        slots += [("R", (x, 88, x + rw, 134)) for x in rx]
    else:
        raise ValueError(ptype)
    return slots


# --------------------------------------------------------------------------- глифы
@lru_cache(maxsize=None)
def _font(idx: int, size: int = 220):
    return ImageFont.truetype(os.path.join(FONT_DIR, FONT_FILES[idx % len(FONT_FILES)]), size)


@lru_cache(maxsize=None)
def _glyph(font_idx: int, ch: str):
    """Маска глифа, обрезанная по контуру, float32 [0,1]."""
    f = _font(font_idx)
    canvas = Image.new("L", (400, 400), 0)
    ImageDraw.Draw(canvas).text((60, 40), ch, fill=255, font=f)
    a = np.asarray(canvas)
    ys, xs = np.where(a > 8)
    g = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255.0
    return g


def _paste_glyph(canvas: np.ndarray, ch: str, box_px, font_idx: int, rng,
                 widen: float, thick: float, color=(0, 0, 0)):
    x0, y0, x1, y1 = box_px
    bw, bh = x1 - x0, y1 - y0
    g = _glyph(font_idx, ch)
    gh, gw = g.shape
    asp = gw / gh
    tw = min(bw, asp * bh * widen)
    if ch == "1":
        tw = min(tw, bw * 0.62)
    tw = max(int(round(tw)), 2)
    th = max(int(round(bh)), 2)
    m = cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA if tw < gw else cv2.INTER_LINEAR)
    if abs(thick) > 1e-3:
        k = max(1, int(round(abs(thick) * th)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        m = cv2.dilate(m, ker) if thick > 0 else cv2.erode(m, ker)
    ox = int(round(x0 + (bw - tw) / 2))
    oy = int(round(y0))
    H, W = canvas.shape[:2]
    xa, ya = max(ox, 0), max(oy, 0)
    xb, yb = min(ox + tw, W), min(oy + th, H)
    if xb <= xa or yb <= ya:
        return
    sub = m[ya - oy:yb - oy, xa - ox:xb - ox][..., None]
    reg = canvas[ya:yb, xa:xb].astype(np.float32)
    col = np.array(color, np.float32)[None, None, :]
    canvas[ya:yb, xa:xb] = (reg * (1 - sub) + col * sub).astype(np.uint8)


def _rounded_mask(w, h, r):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    return np.asarray(m).astype(np.float32) / 255.0


def _bgr(rgb):
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def _jitter(rgb, rng, amt=6):
    return tuple(int(np.clip(c + rng.integers(-amt, amt + 1), 0, 255)) for c in rgb)


def _flag(canvas, x0, y0, x1, y1):
    h = (y1 - y0) / 3
    for i, c in enumerate([(255, 255, 255), (0, 57, 166), (213, 43, 30)]):
        cv2.rectangle(canvas, (int(x0), int(y0 + i * h)), (int(x1), int(y0 + (i + 1) * h)), _bgr(c), -1)
    cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1), int(y1)), (30, 30, 30), 1)


def _text_small(canvas, text, box, font_idx, rng):
    x0, y0, x1, y1 = box
    n = len(text)
    cw = (x1 - x0) / n
    for i, ch in enumerate(text):
        _paste_glyph(canvas, ch, (x0 + i * cw + cw * 0.06, y0, x0 + (i + 1) * cw - cw * 0.06, y1),
                     font_idx, rng, 1.0, 0.0)


# --------------------------------------------------------------------------- знак РФ
def random_text(rng, region_len: int | None = None):
    """Случайный номер A123BC77; возвращает строку без пробелов."""
    L = lambda: LETTERS[rng.integers(len(LETTERS))]
    while True:
        d = "".join(DIGITS[rng.integers(10)] for _ in range(3))
        if d != "000":
            break
    if region_len is None:
        region_len = 3 if rng.random() < 0.18 else 2
    reg = REGIONS3[rng.integers(len(REGIONS3))] if region_len == 3 else REGIONS2[rng.integers(len(REGIONS2))]
    return L() + d + L() + L() + reg


def render_plate(ptype: str, text: str, px_per_mm: float = 2.0, rng=None, font_idx=None,
                 style: dict | None = None):
    """Рисует знак. Возвращает dict(img BGR, alpha, w_mm, h_mm, px_per_mm, slots, text, type).

    slots: список (char, kind, (x0,y0,x1,y1) в пикселях полотна).
    """
    rng = rng or np.random.default_rng()
    st = dict(widen=rng.uniform(0.95, 1.25), thick=rng.uniform(-0.04, 0.0),
              frame=rng.random() < 0.97, rus=rng.random() < 0.95, bg_amt=6)
    if style:
        st.update(style)
    if font_idx is None:
        font_idx = int(rng.integers(len(FONT_FILES)))
    region_len = len(text) - 6
    wmm, hmm = SIZE_MM[ptype]
    s = px_per_mm
    W, H = int(round(wmm * s)), int(round(hmm * s))
    bg = BG_YELLOW if ptype == "type1b" else BG_WHITE
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = _bgr(_jitter(bg, rng, st["bg_amt"]))
    fg = (10, 10, 10)
    thick_line = max(1, int(round(3 * s)))
    inset = int(round(4 * s))
    if st["frame"]:
        r = int(round(8 * s))
        pm = Image.new("L", (W, H), 0)
        ImageDraw.Draw(pm).rounded_rectangle([inset, inset, W - 1 - inset, H - 1 - inset], radius=r,
                                             outline=255, width=thick_line)
        pmask = (np.asarray(pm).astype(np.float32) / 255.0)[..., None]
        canvas = (canvas * (1 - pmask) + np.array(fg, np.float32) * pmask).astype(np.uint8)
    # символы
    chars = text
    lay = layout(ptype, region_len)
    slots = []
    for ch, (kind, box) in zip(chars, lay):
        bpx = tuple(int(round(v * s)) for v in box)
        _paste_glyph(canvas, ch, bpx, font_idx, rng, st["widen"] if kind != "R" else st["widen"] * 0.95,
                     st["thick"], fg)
        slots.append((ch, kind, bpx))
    # разделитель региона и RUS/флаг
    if ptype in ("type1", "type1b"):
        if st["frame"]:
            xs_ = int(round(383 * s))
            cv2.line(canvas, (xs_, int(8 * s)), (xs_, int((hmm - 8) * s)), fg, thick_line)
        if st["rus"]:
            rx0 = 400 if region_len == 2 else 396
            _text_small(canvas, "RUS", (rx0 * s, 80 * s, (rx0 + 62) * s, 100 * s), font_idx, rng)
            _flag(canvas, (rx0 + 70) * s, 82 * s, (rx0 + 100) * s, 99 * s) if region_len == 2 else \
                _flag(canvas, (rx0 + 66) * s, 82 * s, (rx0 + 92) * s, 99 * s)
    else:
        if st["frame"] and rng.random() < 0.6:
            xs_ = int(round(130 * s))
            cv2.line(canvas, (xs_, int(86 * s)), (xs_, int(152 * s)), fg, thick_line)
        if st["rus"]:
            _text_small(canvas, "RUS", (238 * s, 92 * s, 276 * s, 108 * s), font_idx, rng)
            _flag(canvas, 242 * s, 114 * s, 272 * s, 130 * s)
    alpha = _rounded_mask(W, H, int(round(10 * s)))
    return dict(img=canvas, alpha=alpha, w_mm=wmm, h_mm=hmm, px_per_mm=s, slots=slots,
                text=text, type=ptype)


# --------------------------------------------------------------------------- «прочие» знаки
OTHER_KINDS = ["trailer", "moto", "diplomatic", "military", "eu", "us", "by_like", "transit"]


def _plain_plate(wmm, hmm, s, bg_rgb, fg_rgb, lines, rng, font_idx, band=None, border=True, r_mm=10):
    """Универсальный знак: lines = [(text, (x0,y0,x1,y1) мм)]; band — цветная полоса слева."""
    W, H = int(round(wmm * s)), int(round(hmm * s))
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = _bgr(_jitter(bg_rgb, rng, 5))
    fg = _bgr(fg_rgb)
    if band is not None:
        bw, bcol = band
        cv2.rectangle(canvas, (0, 0), (int(bw * s), H), _bgr(bcol), -1)
        for k in range(12):  # кольцо звёзд
            a = 2 * np.pi * k / 12
            cv2.circle(canvas, (int(bw * s / 2 + 0.28 * bw * s * np.cos(a)),
                                int(hmm * s * 0.38 + 0.28 * bw * s * np.sin(a))), max(1, int(1.6 * s)),
                       (0, 215, 255), -1)
    if border:
        pm = Image.new("L", (W, H), 0)
        ImageDraw.Draw(pm).rounded_rectangle([int(3 * s), int(3 * s), W - 1 - int(3 * s), H - 1 - int(3 * s)],
                                             radius=int(7 * s), outline=255, width=max(1, int(2.5 * s)))
        pmask = (np.asarray(pm).astype(np.float32) / 255.0)[..., None]
        canvas = (canvas * (1 - pmask) + np.array(fg, np.float32) * pmask).astype(np.uint8)
    for text, box in lines:
        x0, y0, x1, y1 = [v * s for v in box]
        n = len(text)
        cw = (x1 - x0) / n
        for i, ch in enumerate(text):
            if ch == " ":
                continue
            _paste_glyph(canvas, ch, (x0 + i * cw + cw * 0.05, y0, x0 + (i + 1) * cw - cw * 0.05, y1),
                         font_idx, rng, 1.1, rng.uniform(0, 0.02), fg)
    alpha = _rounded_mask(W, H, int(round(r_mm * s)))
    return canvas, alpha


def render_other(kind: str, px_per_mm: float = 2.0, rng=None):
    """Знаки, которые НЕ являются целевыми (должны получать тип other или не выдаваться)."""
    rng = rng or np.random.default_rng()
    fi = int(rng.integers(len(FONT_FILES)))
    s = px_per_mm
    A = lambda n: "".join(LETTERS[rng.integers(len(LETTERS))] for _ in range(n))
    D = lambda n: "".join(DIGITS[rng.integers(10)] for _ in range(n))
    reg = REGIONS2[rng.integers(len(REGIONS2))]
    if kind == "trailer":        # тип 2: две буквы + 4 цифры + регион, одна строка
        t = A(2) + D(4) + reg
        xs = [(18, 24), (73, 24), (128, 24), (178, 24), (228, 24), (278, 24)]
        lines = [(t[:6], (22, 16, 342, 96)), (t[6:], (362, 16, 500, 76))]
        img, al = _plain_plate(520, 112, s, BG_WHITE, (10, 10, 10), lines, rng, fi)
        w, h = 520, 112
    elif kind == "moto":         # мотоциклетный: 4 цифры + 2 буквы, две строки, 190x145
        t = D(4) + A(2)
        lines = [(t[:4], (20, 16, 170, 72)), (t[4:], (20, 82, 90, 128)), (reg, (100, 82, 170, 118))]
        img, al = _plain_plate(190, 145, s, BG_WHITE, (10, 10, 10), lines, rng, fi)
        w, h = 190, 145
    elif kind == "diplomatic":   # красно-бордовый фон, белые символы: 123 CD 4
        t = D(3) + "CD" + D(1)
        lines = [(t, (30, 16, 360, 96)), (reg, (400, 14, 500, 74))]
        img, al = _plain_plate(520, 112, s, (150, 30, 40), (240, 240, 240), lines, rng, fi)
        w, h = 520, 112
    elif kind == "military":     # 4 цифры + 2 буквы + регион (иная маска), чёрный на белом
        t = D(4) + A(2)
        lines = [(t, (25, 16, 380, 96)), (reg, (400, 14, 500, 74))]
        img, al = _plain_plate(520, 112, s, BG_WHITE, (10, 10, 10), lines, rng, fi)
        w, h = 520, 112
    elif kind == "eu":           # европейский: синяя полоса, одна строка, произвольный текст
        t = A(1 if rng.random() < .5 else 2) + A(2) + D(int(rng.integers(3, 5)))
        lines = [(t, (62, 14, 500, 96))]
        img, al = _plain_plate(520, 112, s, BG_WHITE, (10, 10, 10), lines, rng, fi, band=(48, (20, 60, 160)))
        w, h = 520, 112
    elif kind == "us":           # американский: 305x152, цветной фон
        bg = [(235, 235, 235), (230, 200, 60), (40, 70, 150)][int(rng.integers(3))]
        fg = (10, 10, 10) if bg != (40, 70, 150) else (240, 240, 240)
        t = A(3) + D(4)
        lines = [(t, (22, 40, 283, 120))]
        img, al = _plain_plate(305, 152, s, bg, fg, lines, rng, fi, r_mm=8)
        w, h = 305, 152
    elif kind == "by_like":      # белорусский: 1234 AB-7, красная окантовка-подобная полоса не рисуется
        t = D(4) + A(2) + str(int(rng.integers(1, 8)))
        lines = [(t, (30, 16, 490, 96))]
        img, al = _plain_plate(520, 112, s, BG_WHITE, (10, 10, 10), lines, rng, fi)
        w, h = 520, 112
    else:                        # transit: жёлто-красный
        t = A(2) + D(3) + A(1)
        lines = [(t, (28, 16, 360, 96)), (reg, (400, 14, 500, 74))]
        img, al = _plain_plate(520, 112, s, (235, 235, 235), (200, 30, 30), lines, rng, fi)
        w, h = 520, 112
    return dict(img=img, alpha=al, w_mm=w, h_mm=h, px_per_mm=s, slots=[], text="", type="other",
                other_kind=kind)


def fmt_plate_num(text: str) -> str:
    return text
