"""Генерация кандидатов знаков по группировке символов.

Шаги: адаптивная бинаризация (тёмные символы на светлом фоне) в трёх масштабах ->
компоненты связности, похожие на символы -> цепочки (почти коллинеарные символы
одной высоты) -> гипотезы:
  * одна строка (тип 1 / 1Б): положение символов серии известно (шаг 60 мм),
    четырёхугольник строится по линиям верха и низа символов;
  * две строки (тип 1А): верхняя строка «буква+3 цифры» и нижняя строка
    «2 буквы + регион», четырёхугольник по верху верхней и низу нижней строки.
Точность четырёхугольника невысока; распознаватель уточняет его гомографию
по найденным положениям символов.
"""
import numpy as np
import cv2

SCALES = (1.0, 0.5, 0.25)
BLOCK, CONST = 31, 9


# --------------------------------------------------------------------------- компоненты
def extract_blobs(gray, scale, hmin=9):
    g = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, BLOCK, CONST)
    n, lab, st, cen = cv2.connectedComponentsWithStats(thr, connectivity=8)
    if n <= 1:
        return None
    x, y, w, h, a = st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4]
    H = g.shape[0]
    ratio = h / np.maximum(w, 1)
    fill = a / np.maximum(w * h, 1)
    m = (h >= hmin) & (h <= 0.4 * H) & (w >= 2) & (ratio >= 0.9) & (ratio <= 7.0) & (fill >= 0.18) & (fill <= 0.95)
    m[0] = False
    ids = np.nonzero(m)[0]
    if len(ids) < 3:
        return None
    if len(ids) > 900:
        ids = ids[np.argsort(-h[ids])[:900]]
    return dict(ids=ids, x=x[ids], y=y[ids], w=w[ids], h=h[ids], cx=cen[ids, 0], cy=cen[ids, 1],
                lab=lab, scale=scale)


def _components(B):
    n = len(B["ids"])
    cx, cy, h = B["cx"], B["cy"], B["h"].astype(np.float64)
    dx = cx[None, :] - cx[:, None]
    dy = cy[None, :] - cy[:, None]
    dist = np.hypot(dx, dy)
    hm = np.maximum(h[None, :], h[:, None])
    hr = np.minimum(h[None, :], h[:, None]) / hm
    ok = (dist < 1.9 * hm) & (dist > 0.25 * hm) & (hr > 0.55) & (np.abs(dy) <= 0.8 * np.abs(dx) + 0.25 * hm)
    ok = np.triu(ok, 1)
    ii, jj = np.nonzero(ok)
    adj = [[] for _ in range(n)]
    for i, j in zip(ii, jj):
        adj[i].append(j); adj[j].append(i)
    seen = np.zeros(n, bool)
    comps = []
    for s in range(n):
        if seen[s] or not adj[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop(); comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True; stack.append(v)
        if len(comp) >= 3:
            comps.append(comp)
    return comps, (ii, jj)


def _extract_chains(B, comp, edges_set, rng):
    cx, cy, h = B["cx"], B["cy"], B["h"].astype(np.float64)
    remaining = list(comp)
    chains = []
    comp_set = set(comp)
    while len(remaining) >= 3:
        rem = np.array(remaining)
        cand_edges = [(i, j) for (i, j) in edges_set if i in comp_set and j in comp_set]
        cand_edges = [(i, j) for (i, j) in cand_edges if i in remaining and j in remaining] if len(remaining) < len(comp) else cand_edges
        if not cand_edges:
            break
        if len(cand_edges) > 70:
            sel = rng.choice(len(cand_edges), 70, replace=False)
            cand_edges = [cand_edges[k] for k in sel]
        best = None
        for (i, j) in cand_edges:
            p = np.array([cx[i], cy[i]]); d = np.array([cx[j] - cx[i], cy[j] - cy[i]])
            nd = np.linalg.norm(d)
            if nd < 1e-6:
                continue
            d /= nd
            rel = np.stack([cx[rem] - p[0], cy[rem] - p[1]], 1)
            perp = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
            hh = 0.5 * (h[i] + h[j])
            hr = np.minimum(h[rem], hh) / np.maximum(h[rem], hh)
            inl = rem[(perp < 0.32 * hh) & (hr > 0.55)]
            if best is None or len(inl) > len(best):
                best = inl
        if best is None or len(best) < 3:
            break
        chains.append(list(best))
        bs = set(best.tolist())
        remaining = [r for r in remaining if r not in bs]
    return chains


def _split_segments(B, chain):
    """Сортирует цепочку вдоль её направления и режет по большим разрывам."""
    cx, cy, h = B["cx"][chain], B["cy"][chain], B["h"][chain].astype(np.float64)
    P = np.stack([cx, cy], 1)
    c = P - P.mean(0)
    u = np.linalg.svd(c, full_matrices=False)[2][0]
    if u[0] < 0:
        u = -u
    a = P @ u
    order = np.argsort(a)
    chain = np.array(chain)[order]; a = a[order]; hs = h[order]
    segs, cur = [], [0]
    for k in range(1, len(chain)):
        gap = a[k] - a[k - 1]
        if gap > 1.9 * max(hs[k], hs[k - 1]):
            segs.append(cur); cur = [k]
        else:
            cur.append(k)
    segs.append(cur)
    return [chain[s] for s in segs if len(s) >= 3], u


# --------------------------------------------------------------------------- геометрия
def _blob_extents(B, k, u, n):
    """Крайние точки компоненты вдоль n (верх/низ) и вдоль u (лево/право)."""
    x0, y0, w, h = int(B["x"][k]), int(B["y"][k]), int(B["w"][k]), int(B["h"][k])
    ys, xs = np.nonzero(B["lab"][y0:y0 + h, x0:x0 + w] == B["ids"][k])
    px = xs + x0 + 0.5
    py = ys + y0 + 0.5
    pn = px * n[0] + py * n[1]
    pu = px * u[0] + py * u[1]
    return pn.min(), pn.max(), pu.min(), pu.max()


def _fit_pitch(a, h_ref, pitch_ratio, kmax):
    """Подбор шага символов: возвращает (p, k_i, остаток)."""
    a0 = a[0]
    best = None
    for f in np.arange(0.40, 1.25, 0.01):
        p = f * pitch_ratio * h_ref
        k = np.rint((a - a0) / p).astype(int)
        if len(set(k.tolist())) != len(k) or k.max() > kmax:
            continue
        r = float(np.sum((a - a0 - k * p) ** 2)) / (p * p)
        if best is None or r < best[2] - 1e-6:
            best = (p, k, r)
    return best


def _line_fit(a, v):
    if len(a) >= 2 and np.ptp(a) > 1e-6:
        return np.polyfit(a, v, 1)
    return np.array([0.0, float(np.mean(v))])


def single_row_hyps(B, seg, u):
    """Гипотезы четырёхугольника для одной строки (тип 1/1Б). Координаты — в системе B."""
    n = np.array([-u[1], u[0]])
    if n[1] < 0:
        n = -n
    seg = list(seg)
    hs = B["h"][seg].astype(np.float64)
    # реальные знаки: буквы и цифры региона ≈0.83 высоты цифр серии, поэтому порог мягкий
    tall = [k for k, t in zip(seg, hs >= 0.72 * hs.max()) if t]
    if len(tall) < 3:
        return []
    ext = {k: _blob_extents(B, k, u, n) for k in seg}
    tall.sort(key=lambda k: ext[k][2] + ext[k][3])
    A_all = np.array([0.5 * (ext[k][2] + ext[k][3]) for k in tall])
    top_all = np.array([ext[k][0] for k in tall]); bot_all = np.array([ext[k][1] for k in tall])
    hyps = []
    for m in range(min(len(tall), 6), 2, -1):
        a_t = A_all[:m]; top = top_all[:m]; bot = bot_all[:m]
        h_ref = float(np.percentile(bot - top, 75))
        fit = _fit_pitch(a_t, h_ref, 0.75, 5)
        if fit is None:
            continue
        p, kk, res = fit
        if res > 0.35 * m:
            continue
        rest = len(tall) - m
        reg = [k for k in seg if k not in tall[:m] and 0.4 * h_ref < B["h"][k] < 1.0 * h_ref
               and 0.5 * (ext[k][2] + ext[k][3]) > a_t[-1] + 0.3 * p] if rest >= 0 else []
        hyps.append((m, a_t, top, bot, h_ref, p, kk, res, reg))
        if len(hyps) >= 1:
            break
    out = []
    for (m, a_t, top, bot, h_ref, p, kk, res, reg) in hyps:
        out += _single_row_from(a_t, top, bot, h_ref, p, kk, res, reg, u, n, len(tall))
    return out


def _single_row_from(a_t, top, bot, h_ref, p, kk, res, reg, u, n, n_tall):
    tall = a_t
    offs = []
    if len(reg) >= 1:
        offs.append(5 - int(kk.max()))
    else:
        offs.append(0)
        if int(kk.max()) < 5:
            offs.append(5 - int(kk.max()))
    bot_fit = _line_fit(a_t, bot)
    top_fit = np.array([bot_fit[0], float(np.percentile(top - bot_fit[0] * a_t, 30))])
    s_h = p / 60.0
    out = []
    for o in offs:
        if o < 0:
            continue
        a_c0 = a_t[0] - o * p                                    # центр символа №0 (индекс в номере)
        aL = a_c0 - 50 * s_h
        aR = a_c0 + (520 - 50) * s_h
        quad = []
        for aa, which in ((aL, "t"), (aR, "t"), (aR, "b"), (aL, "b")):
            t_ = np.polyval(top_fit, aa); b_ = np.polyval(bot_fit, aa)
            sv = (b_ - t_) / 80.0
            nn = t_ - 16 * sv if which == "t" else b_ + 16 * sv
            quad.append(aa * u + nn * n)
        quad = np.array(quad, np.float32)
        out.append(dict(quad=quad, ptype="type1", n_tall=len(a_t), n_reg=max(len(reg), n_tall - len(a_t)), res=res / max(len(a_t), 1),
                        h=h_ref))
    return out


def two_row_hyps(B, segA, uA, segB, uB):
    """Гипотеза для типа 1А: segA — верхняя строка, segB — нижняя."""
    if abs(float(np.dot(uA, uB))) < np.cos(np.radians(11)):
        return []
    u = uA if np.dot(uA, uB) > 0 else -uA
    n = np.array([-u[1], u[0]])
    if n[1] < 0:
        n = -n
    hA = B["h"][segA].astype(np.float64); hB = B["h"][segB].astype(np.float64)
    tallA = [k for k, h in zip(segA, hA) if h >= 0.82 * hA.max()]
    tallB = [k for k, h in zip(segB, hB) if h >= 0.82 * hB.max()]
    if len(tallA) < 3 or len(tallB) < 2:
        return []
    extA = {k: _blob_extents(B, k, u, n) for k in tallA}
    extB = {k: _blob_extents(B, k, u, n) for k in tallB}
    aA = np.array([0.5 * (extA[k][2] + extA[k][3]) for k in tallA]); order = np.argsort(aA)
    tallA = [tallA[i] for i in order]; aA = aA[order]
    aB = np.array([0.5 * (extB[k][2] + extB[k][3]) for k in tallB]); order = np.argsort(aB)
    tallB = [tallB[i] for i in order]; aB = aB[order]
    topA = np.array([extA[k][0] for k in tallA]); botA = np.array([extA[k][1] for k in tallA])
    topB = np.array([extB[k][0] for k in tallB]); botB = np.array([extB[k][1] for k in tallB])
    h_ref = float(np.median(botA - topA))
    # вертикальное расстояние строк: верх нижней строки - верх верхней (в норме 74 мм при высоте 62 мм)
    dv = float(np.median(topB) - np.median(topA))
    if not (0.9 * h_ref < dv < 1.6 * h_ref):
        return []
    fit = _fit_pitch(aA, h_ref, 48.0 / 62.0, 3)
    if fit is None:
        return []
    p, kk, res = fit
    if res > 0.35 * len(tallA):
        return []
    # индекс первого символа верхней строки определяем по левому символу нижней строки (оба — столбец №0)
    o = int(np.clip(round((aA[0] - aB[0]) / p), 0, 3)) if abs(aA[0] - aB[0]) > 0.5 * p else 0
    if o < 0:
        o = 0
    top_fit = _line_fit(aA, topA)
    bot_fit = _line_fit(aB, botB)
    s_h = p / 48.0
    a_c0 = aA[0] - (kk[0] + o) * p if False else aA[0] - o * p
    aL = a_c0 - 50 * s_h
    aR = a_c0 + (290 - 50) * s_h
    quad = []
    for aa, which in ((aL, "t"), (aR, "t"), (aR, "b"), (aL, "b")):
        t_ = np.polyval(top_fit, aa); b_ = np.polyval(bot_fit, aa)
        sv = (b_ - t_) / 136.0
        nn = t_ - 14 * sv if which == "t" else b_ + 20 * sv
        quad.append(aa * u + nn * n)
    return [dict(quad=np.array(quad, np.float32), ptype="type1a", n_tall=len(tallA) + len(tallB), n_reg=0,
                 res=res / len(tallA), h=h_ref)]


# --------------------------------------------------------------------------- второй источник: плотность границ
def edge_candidates(gray, scales=(1.0, 0.5), max_per_scale=6):
    """Прямоугольники с плотными границами (устойчивее к смазу и слипанию символов).

    Возвращает гипотезы с квадрантом, слегка раздутым относительно найденной области символов.
    """
    out = []
    for sc in scales:
        g = gray if sc == 1.0 else cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        gx = cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=3)
        mag = cv2.convertScaleAbs(np.abs(gx) + 0.6 * np.abs(gy), alpha=0.25)
        _, bw = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = max(5, int(0.012 * g.shape[1]) | 1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuralElement(cv2.MORPH_RECT, (k * 2 + 1, k))
                              if hasattr(cv2, "getStructuralElement") else cv2.getStructuringElement(cv2.MORPH_RECT, (k * 2 + 1, k)))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 250:
                continue
            (cx, cy), (w, h), ang = cv2.minAreaRect(c)
            if w < h:
                w, h, ang = h, w, ang + 90
            if h < 8 or w < 28:
                continue
            asp = w / h
            fill = area / (w * h)
            if fill < 0.5:
                continue
            if 2.4 <= asp <= 7.0:
                pt = "type1"
            elif 1.25 <= asp < 2.4:
                pt = "type1a"
            else:
                continue
            found.append((area * fill, cx, cy, w, h, ang, pt))
        found.sort(reverse=True)
        for _, cx, cy, w, h, ang, pt in found[:max_per_scale]:
            ex = (1.14, 1.28) if pt == "type1" else (1.20, 1.18)
            box = cv2.boxPoints(((cx, cy), (w * ex[0], h * ex[1]), ang)).astype(np.float32)
            # порядок TL, TR, BR, BL относительно длинной стороны
            c0 = box.mean(0)
            d = box - c0
            ang_ = np.arctan2(d[:, 1], d[:, 0])
            box = box[np.argsort(ang_)]                 # по часовой от «правого-верхнего» в системе экрана
            # переставить так, чтобы первая точка была левой-верхней
            idx = int(np.argmin(box[:, 0] + box[:, 1]))
            box = np.roll(box, -idx, axis=0)
            if np.linalg.norm(box[1] - box[0]) < np.linalg.norm(box[3] - box[0]):
                box = np.roll(box, -1, axis=0)
            out.append(dict(quad=box / sc, ptype=pt, n_tall=5, n_reg=0, res=0.5, h=h / sc, scale=sc, src="edge",
                            prior=2.0))
    return out


# --------------------------------------------------------------------------- вход
def candidates(gray, rng=None, scales=SCALES, max_cands=40, use_edge=False, upscale=2.0):
    """Возвращает список гипотез dict(quad (в координатах gray), ptype, prior)."""
    rng = rng or np.random.default_rng(0)
    out = []
    scales = list(scales)
    for idx in range(len(scales) + 1):
        if idx < len(scales):
            sc = scales[idx]
        else:
            # каскад: мелкие знаки находятся только на увеличенном изображении — лишь если надёжного кандидата нет
            if not upscale or max(gray.shape) * upscale > 2600:
                break
            sc = upscale
        B = extract_blobs(gray, sc)
        if B is None:
            continue
        comps, (ii, jj) = _components(B)
        edges = set(zip(ii.tolist(), jj.tolist()))
        for comp in comps[:60]:
            chains = _extract_chains(B, comp, edges, rng)
            segs = []
            for ch in chains:
                for seg, u in _split_segments(B, ch)[0:1] if False else [(s, _split_segments(B, ch)[1]) for s in _split_segments(B, ch)[0]]:
                    segs.append((seg, u))
            for seg, u in segs:
                if len(seg) >= 5:
                    for h in single_row_hyps(B, seg, u):
                        h["scale"] = sc; h["quad"] = h["quad"] / sc; out.append(h)
            # пары строк для типа 1А
            for ia, (sa, ua) in enumerate(segs):
                for ib, (sb, ub) in enumerate(segs):
                    if ia == ib or len(sa) < 3 or len(sb) < 3:
                        continue
                    for h in two_row_hyps(B, sa, ua, sb, ub):
                        h["scale"] = sc; h["quad"] = h["quad"] / sc; out.append(h)
    # приоритет: больше высоких символов, меньше остаток подгонки
    for h in out:
        h["prior"] = h["n_tall"] + 1.5 * (h["n_reg"] > 0) - 3.0 * h["res"]
    if use_edge:
        out += edge_candidates(gray)
    out.sort(key=lambda d: -d["prior"])
    # убираем почти одинаковые четырёхугольники
    kept = []
    for h in out:
        c = h["quad"].mean(0)
        wd = np.linalg.norm(h["quad"][1] - h["quad"][0])
        dup = False
        for k in kept:
            if k["ptype"] == h["ptype"] and np.linalg.norm(k["quad"].mean(0) - c) < 0.15 * wd:
                dup = True
                break
        if not dup:
            kept.append(h)
        if len(kept) >= max_cands:
            break
    return kept
