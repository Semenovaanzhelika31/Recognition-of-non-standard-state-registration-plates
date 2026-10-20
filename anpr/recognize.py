"""Выпрямление, распознавание по слотам ГОСТ с декодированием по маске номера.

Маска: [БУКВА][ЦИФРА][ЦИФРА][ЦИФРА][БУКВА][БУКВА][КОД РЕГИОНА 2–3 цифры],
трёхзначный код начинается с 1, 2 или 7. Положения слотов известны из геометрии
знака; классификатор оценивает окна со сдвигами, Витерби выбирает согласованные
сдвиги (знак — жёсткий объект), затем гомография уточняется по найденным
положениям символов и распознавание повторяется.
"""
import cv2
import numpy as np

from . import gost
from .features import cut_windows, featurize
from .gost import DIGIT_IDX, LETTER_IDX, N_CLS, NONE_CLS

ALPHA = gost.ALPHABET
FIRST3 = [gost.CHAR_TO_CLS[c] for c in "127"]


class MLP:
    """Многослойный перцептрон (веса из npz, только numpy)."""

    def __init__(self, path):
        z = np.load(path)
        n = int(z["n_layers"])
        self.W = [z[f"W{i}"].astype(np.float32) for i in range(n)]
        self.b = [z[f"b{i}"].astype(np.float32) for i in range(n)]
        self.mu = z["mu"].astype(np.float32) if "mu" in z else None
        self.sd = z["sd"].astype(np.float32) if "sd" in z else None

    def logits(self, x):
        h = x.astype(np.float32)
        if self.mu is not None:
            h = (h - self.mu) / self.sd
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.maximum(h @ W + b, 0)
        return h @ self.W[-1] + self.b[-1]

    def proba(self, x):
        z = self.logits(x)
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)


# --------------------------------------------------------------------------- выпрямление
def homography_from_quad(quad, ptype):
    return cv2.getPerspectiveTransform(np.asarray(quad, np.float32), gost.canon_corners(ptype))


def warp_to_canon(img, M, ptype):
    cw, ch = gost.canon_size(ptype)
    # оценка масштаба: сколько пикселей кадра на пиксель канона по горизонтали
    Mi = np.linalg.inv(M)
    p = Mi @ np.array([[0, cw], [0, 0], [1, 1]], np.float64)
    wq = abs(p[0, 1] / p[2, 1] - p[0, 0] / p[2, 0])
    src = img
    factor = cw / max(wq, 1.0)
    if factor < 0.7:
        src = cv2.GaussianBlur(img, (0, 0), 0.45 / factor)
    return cv2.warpPerspective(src, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------- декодирование
def _slot_groups(ptype, region_len):
    """Индексы слотов по цепочкам Витерби."""
    n = 6 + region_len if ptype != "type1a" else 6 + region_len
    if ptype == "type1a":
        return [list(range(0, 4)), list(range(4, 6 + region_len))]
    return [list(range(0, 6 + region_len))]


def _eval_slots(gray, slots, rec, dx_f, dy_f):
    wins = []
    for kind, box in slots:
        bw, bh = box[2] - box[0], box[3] - box[1]
        shifts = [(dx * bw, dy * bh) for dx in dx_f for dy in dy_f]
        wins.append(cut_windows(gray, box, shifts))
    P = rec.proba(featurize(np.concatenate(wins)))
    return np.log(P.reshape(len(slots), len(dx_f), len(dy_f), N_CLS) + 1e-6)


def _viterbi(L, slots, groups, dx_f, region_len, lam):
    """L: [S,nx,ny,C]. Возвращает (sel, total, mean) где sel[s]=(ix,iy,cls)."""
    nx = len(dx_f)
    pen = lam * (dx_f[:, None] - dx_f[None, :]) ** 2
    sel = {}
    total = 0.0
    for g in groups:
        emit, arg = [], []
        for pos, s in enumerate(g):
            kind = slots[s][0]
            if kind == "L":
                allowed = LETTER_IDX
            elif kind == "D":
                allowed = DIGIT_IDX
            else:
                first_r = [t for t in g if slots[t][0] == "R"][0]
                allowed = FIRST3 if (region_len == 3 and s == first_r) else DIGIT_IDX
            sub = L[s][:, :, allowed]                      # nx, ny, A
            A = len(allowed)
            flat = sub.reshape(nx, -1)
            j = flat.argmax(1)
            emit.append(flat.max(1))
            arg.append(np.stack([j // A, np.array(allowed)[j % A]], 1))
        n = len(g)
        score = emit[0].copy()
        back = np.zeros((n, nx), int)
        for t in range(1, n):
            cand = score[:, None] - pen                  # [prev, cur]
            back[t] = cand.argmax(0)
            score = cand.max(0) + emit[t]
        ix = int(score.argmax())
        total += float(score[ix])
        for t in range(n - 1, -1, -1):
            iy, c = arg[t][ix]
            sel[g[t]] = (ix, int(iy), int(c))
            ix = back[t][ix]
    return sel, total


def decode(gray, ptype, rec, dx_f, dy_f, lam=10.0, thr=0.2):
    """Декодирует выпрямленное изображение для обеих гипотез длины региона.

    Возвращает список вариантов (по убыванию качества).
    """
    base = gost.canon_slots(ptype, 2)
    base_main = [s for s in base if s[0] != "R"]
    r2 = [s for s in base if s[0] == "R"]
    r3 = [s for s in gost.canon_slots(ptype, 3) if s[0] == "R"]
    L_main = _eval_slots(gray, base_main, rec, dx_f, dy_f)
    out = []
    for rl, rs in ((2, r2), (3, r3)):
        slots = base_main + rs
        L = np.concatenate([L_main, _eval_slots(gray, rs, rec, dx_f, dy_f)], 0)
        groups = _slot_groups(ptype, rl)
        sel, total = _viterbi(L, slots, groups, dx_f, rl, lam)
        S = len(slots)
        chars, conf, shifts, lp = [], [], [], []
        for s in range(S):
            ix, iy, c = sel[s]
            p = float(np.exp(L[s, ix, iy, c]))
            conf.append(p)
            lp.append(float(L[s, ix, iy, c]))
            chars.append(ALPHA[c] if p >= thr else "#")
            bw, bh = slots[s][1][2] - slots[s][1][0], slots[s][1][3] - slots[s][1][1]
            shifts.append((dx_f[ix] * bw, dy_f[iy] * bh))
        out.append(dict(region_len=rl, text="".join(chars), conf=np.array(conf), shifts=shifts, slots=slots,
                        mean_lp=float(np.mean(lp)), min_lp=float(np.min(lp)), total=total / S,
                        pnone=float(np.mean([np.exp(L[s, sel[s][0], sel[s][1], NONE_CLS]) for s in range(S)]))))
    out.sort(key=lambda d: -(d["total"] + (0.03 if d["region_len"] == 2 else 0.0)))
    return out


def refine_homography(M, var, ptype):
    """Корректирует M (кадр->канон) по найденным сдвигам символов; None, если коррекция сомнительна."""
    src, dst = [], []
    for (kind, box), (dx, dy), c in zip(var["slots"], var["shifts"], var["conf"]):
        if c < 0.3:
            continue
        x0, y0, x1, y1 = box
        for (px, py) in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            dst.append((px, py))
            src.append((px + dx, py + dy))
    if len(src) < 12:
        return None
    src, dst = np.array(src, np.float32), np.array(dst, np.float32)
    T, _ = cv2.findHomography(src, dst, cv2.LMEDS)
    if T is None:
        return None
    cw, ch = gost.canon_size(ptype)
    pts = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float64)
    q = (T @ np.vstack([pts.T, np.ones(4)]))
    q = (q[:2] / q[2]).T
    if np.abs(q - pts).max() > 0.3 * max(cw, ch) or not np.isfinite(q).all():
        return None
    return T @ M


def plate_color_stats(bgr_can, ptype, slots):
    """Статистики фона знака: яркость, насыщенность, жёлтость, синяя полоса слева."""
    h, w = bgr_can.shape[:2]
    gray = cv2.cvtColor(bgr_can, cv2.COLOR_BGR2GRAY)
    mask = np.ones((h, w), bool)
    for _, (x0, y0, x1, y1) in slots:
        mask[max(int(y0) - 2, 0):int(y1) + 3, max(int(x0) - 2, 0):int(x1) + 3] = False
    sx = gost.CAN_SCALE[ptype]
    m = int(8 * sx)
    mask[:m, :] = False; mask[-m:, :] = False; mask[:, :m] = False; mask[:, -m:] = False
    bg = bgr_can[mask] if mask.sum() > 20 else bgr_can.reshape(-1, 3)
    g = gray[mask] if mask.sum() > 20 else gray.reshape(-1)
    lvl = float(np.median(g))
    # пиксели «фона» — верхняя половина по яркости (исключает RUS/линии)
    sel = bg[g >= np.percentile(g, 40)]
    med = np.median(sel, 0)
    hsv = cv2.cvtColor(med.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
    hue, sat = float(hsv[0]), float(hsv[1]) / 255.0
    left = bgr_can[:, :max(2, int(0.07 * w))].reshape(-1, 3).astype(np.float32).mean(0)
    blue_left = float((left[0] - left[2]) / 255.0)
    # контраст «чернила»: тёмные пиксели внутри слотов
    ink = []
    for _, (x0, y0, x1, y1) in slots:
        ink.append(gray[int(y0):int(y1), int(x0):int(x1)].reshape(-1))
    ink = np.concatenate(ink) if ink else gray.reshape(-1)
    dark = float(np.percentile(ink, 10))
    return dict(level=lvl / 255.0, sat=sat, hue=hue, blue_left=blue_left,
                contrast=(lvl - dark) / 255.0, bg_std=float(np.std(g)) / 255.0)


def is_yellow(st):
    return st["sat"] > 0.38 and 12 <= st["hue"] <= 40


DX1 = np.arange(-1.125, 1.13, 0.375)   # сдвиги первого прохода (в долях слота): широкий поиск исправляет ошибку детектора на ~1 символ


def read_candidate(gray, bgr, quad, ptype, rec, fine=True):
    """Полное распознавание кандидата. Возвращает dict или None."""
    M = homography_from_quad(quad, ptype)
    C = warp_to_canon(gray, M, ptype)
    dx1 = DX1
    vars1 = decode(C, ptype, rec, dx1, np.array([0.0]))
    best1 = vars1[0]
    M2, passes = M, 1
    res = best1
    if fine:
        cand_vars = [v for v in vars1 if v["total"] > best1["total"] - 0.3][:2]
        best = None
        dx2 = np.array([-0.14, -0.07, 0.0, 0.07, 0.14])
        dy2 = np.array([-0.06, 0.0, 0.06])
        for v in cand_vars:
            Mn = refine_homography(M, v, ptype)
            if Mn is None:
                Mn = M
            C2 = warp_to_canon(gray, Mn, ptype)
            v2 = decode(C2, ptype, rec, dx2, dy2)
            v2 = [x for x in v2 if x["region_len"] == v["region_len"]][0]
            # второй уточняющий шаг
            Mn2 = refine_homography(Mn, v2, ptype)
            if Mn2 is not None:
                C3 = warp_to_canon(gray, Mn2, ptype)
                v3 = decode(C3, ptype, rec, dx2, dy2)
                v3 = [x for x in v3 if x["region_len"] == v["region_len"]][0]
                if v3["total"] > v2["total"]:
                    v2, Mn, C2 = v3, Mn2, C3
            v2["M"], v2["C"] = Mn, C2
            if best is None or v2["total"] > best["total"]:
                best = v2
        res = best
        M2 = res["M"]
        passes = 2
    else:
        res["M"], res["C"] = M, C
    bgr_can = warp_to_canon(bgr, res["M"], ptype)
    st = plate_color_stats(bgr_can, ptype, res["slots"])
    res["stats"] = st
    res["ptype"] = ptype
    # квадрат знака в кадре после уточнения
    cw, ch = gost.canon_size(ptype)
    Mi = np.linalg.inv(res["M"])
    pts = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float64).T
    q = Mi @ np.vstack([pts, np.ones(4)])
    res["quad"] = (q[:2] / q[2]).T.astype(np.float32)
    return res
