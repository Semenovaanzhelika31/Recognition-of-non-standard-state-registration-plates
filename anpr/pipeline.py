"""Полный пайплайн: изображение -> список знаков (текст, тип, уверенность)."""
import os

import cv2
import numpy as np

from . import detect, gost
from .recognize import MLP, is_yellow, read_candidate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
FEAT_NAMES = ["mean_lp", "min_lp", "frac_p50", "frac_p20", "pnone", "region3", "is_1a", "log_w", "level", "sat",
              "yellow", "blue_left", "contrast", "bg_std", "adx", "ady", "n_tall", "has_reg", "fit_res", "total"]


def poly_iou(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    ia, _ = cv2.intersectConvexConvex(a, b)
    ua = cv2.contourArea(a) + cv2.contourArea(b) - ia
    return float(ia / ua) if ua > 1e-6 else 0.0


def verifier_features(res, cand, wpx):
    st = res["stats"]
    conf = res["conf"]
    adx = float(np.mean([abs(s[0]) / (b[2] - b[0]) for s, (_, b) in zip(res["shifts"], res["slots"])]))
    ady = float(np.mean([abs(s[1]) / (b[3] - b[1]) for s, (_, b) in zip(res["shifts"], res["slots"])]))
    return np.array([res["mean_lp"], res["min_lp"], float((conf > 0.5).mean()), float((conf > 0.2).mean()),
                     res["pnone"], float(res["region_len"] == 3), float(res["ptype"] == "type1a"),
                     np.log(max(wpx, 1.0)), st["level"], st["sat"], float(is_yellow(st)), st["blue_left"],
                     st["contrast"], st["bg_std"], adx, ady, cand["n_tall"] / 9.0, float(cand["n_reg"] > 0),
                     cand["res"], res["total"]], np.float32)


class PlateReader:
    def __init__(self, models_dir=MODELS, max_work=1280, max_rec=2000, max_cands=7, prune=-1.2,
                 conf_thr=None, verbose=False):
        self.rec = MLP(os.path.join(models_dir, "recognizer.npz"))
        vp = os.path.join(models_dir, "verifier.npz")
        self.ver = MLP(vp) if os.path.exists(vp) else None
        meta = os.path.join(models_dir, "verifier_threshold.txt")
        self.conf_thr = conf_thr if conf_thr is not None else (float(open(meta).read()) if os.path.exists(meta) else 0.5)
        self.max_work, self.max_rec, self.max_cands, self.prune = max_work, max_rec, max_cands, prune
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ анализ
    def analyze(self, bgr, prune=None, max_cands=None):
        """Возвращает все распознанные кандидаты (с признаками), без порога и NMS."""
        prune = self.prune if prune is None else prune
        max_cands = max_cands or self.max_cands
        h0, w0 = bgr.shape[:2]
        s_work = min(1.0, self.max_work / max(h0, w0))
        s_rec = min(1.0, self.max_rec / max(h0, w0))
        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if s_rec < 1.0:
            bgr_r = cv2.resize(bgr, None, fx=s_rec, fy=s_rec, interpolation=cv2.INTER_AREA)
            gray_r = cv2.resize(gray_full, None, fx=s_rec, fy=s_rec, interpolation=cv2.INTER_AREA)
        else:
            bgr_r, gray_r = bgr, gray_full
        gray_w = gray_r if abs(s_work - s_rec) < 1e-6 else cv2.resize(gray_full, None, fx=s_work, fy=s_work,
                                                                       interpolation=cv2.INTER_AREA)
        k = s_rec / s_work
        cands = detect.candidates(gray_w, self.rng)[:max_cands]
        out = []
        for c in cands:
            quad = (c["quad"] * k).astype(np.float32)
            wq = float(np.linalg.norm(quad[1] - quad[0]))
            if wq < 20 or not np.isfinite(quad).all():
                continue
            # быстрый первый проход, затем уточнение
            q1 = read_candidate(gray_r, bgr_r, quad, c["ptype"], self.rec, fine=False)
            if q1["total"] < prune:
                continue
            res = read_candidate(gray_r, bgr_r, quad, c["ptype"], self.rec, fine=True)
            res["feat"] = verifier_features(res, c, wq)
            res["quad_img"] = res["quad"] / s_rec          # координаты исходного изображения
            res["cand"] = c
            out.append(res)
        return out

    # ------------------------------------------------------------------ вывод
    def score(self, res):
        f = res["feat"][None]
        if self.ver is not None:
            z = self.ver.logits(f)[0]
            return float(1.0 / (1.0 + np.exp(-(z[1] - z[0])))) if len(z) == 2 else float(1 / (1 + np.exp(-z[0])))
        return float(np.exp(res["mean_lp"]))

    def read(self, bgr, emit_other=False):
        """Возвращает [{plate_num, plate_type, confidence, quad}], уже после порога и NMS."""
        res = self.analyze(bgr)
        items = []
        for r in res:
            st = r["stats"]
            yellow = is_yellow(st)
            if r["ptype"] == "type1a":
                if yellow:
                    continue                      # жёлтый двухстрочный — не целевой
                ptype = "type1a"
            else:
                ptype = "type1b" if yellow else "type1"
            conf = self.score(r)
            items.append(dict(plate_num=r["text"], plate_type=ptype, confidence=conf, quad=r["quad_img"]))
        items.sort(key=lambda d: -d["confidence"])
        kept = []
        for it in items:
            if it["confidence"] < self.conf_thr:
                continue
            if any(poly_iou(it["quad"], k["quad"]) > 0.25 for k in kept):
                continue
            kept.append(it)
        return kept
