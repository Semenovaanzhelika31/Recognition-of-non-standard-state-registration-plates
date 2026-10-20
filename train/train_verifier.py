#!/usr/bin/env python3
"""Обучение верификатора кандидатов (оценка уверенности) и подбор порога.

Прогоняет пайплайн по свежим синтетическим сценам (seed не пересекается с датасетом
и тестом), размечает каждого кандидата: 1 — совпал по положению (IoU>=0.4) с целевым знаком
на транспортном средстве и номер прочитан ТОЧНО верно, 0 — иначе. Знаки на щитах
(is_vehicle=0) в обучении пропускаются. Порог выбирается по максимуму F1 на отложенной части.

    python train/train_verifier.py --n 1500 --seed 31 --out models
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anpr.pipeline import PlateReader, poly_iou  # noqa: E402
from anpr.recognize import is_yellow  # noqa: E402
from train.train_recognizer import export_mlp  # noqa: E402
import scenes as sc  # noqa: E402

_reader = None


def _init(models):
    global _reader
    _reader = PlateReader(models)


def work(args):
    idx, seed = args
    rng = np.random.default_rng(np.random.SeedSequence([seed, idx]))
    W, H = [(640, 480), (800, 600), (960, 540), (1280, 720)][rng.choice(4, p=[.4, .25, .2, .15])]
    img, plates = sc.make_scene(rng, W, H)
    tg = [p for p in plates if p["type"] in ("type1", "type1a", "type1b") and p["is_vehicle"] == 1]
    ig = [p for p in plates if p["is_vehicle"] == 0]
    res = _reader.analyze(img, prune=-9.0, max_cands=20)
    X, y, meta = [], [], []
    for r in res:
        q = r["quad_img"]
        if any(poly_iou(q, g["quad"]) > 0.4 for g in ig) and not any(poly_iou(q, g["quad"]) > 0.4 for g in tg):
            continue
        st = r["stats"]
        yel = is_yellow(st)
        if r["ptype"] == "type1a":
            ptype = "type1a" if not yel else "x"
        else:
            ptype = "type1b" if yel else "type1"
        lab = 0
        for g in tg:
            if poly_iou(q, g["quad"]) > 0.4 and r["text"] == g["text"] and ptype == g["type"]:
                lab = 1
        X.append(r["feat"]); y.append(lab)
        meta.append(r["text"])
    return X, y, len(tg), sum(1 for p in tg if p["scale"] >= 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models"))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.models,)) as pool:
        res = pool.map(work, [(i, a.seed) for i in range(a.n)], chunksize=10)
    X = np.array([f for r in res for f in r[0]], np.float32)
    y = np.array([l for r in res for l in r[1]])
    n_gt = sum(r[2] for r in res)
    print("candidates", len(y), "positives", int(y.sum()), "targets in scenes", n_gt, "t=%.0fs" % (time.time() - t0))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, va = perm[:cut], perm[cut:]
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(48, 24), alpha=1e-3, max_iter=400, early_stopping=False, random_state=0)
    clf.fit((X[tr] - mu) / sd, y[tr])
    p = clf.predict_proba((X[va] - mu) / sd)[:, 1]
    best = (0, 0.5)
    for thr in np.linspace(0.2, 0.9, 36):
        tp = ((p >= thr) & (y[va] == 1)).sum(); fp = ((p >= thr) & (y[va] == 0)).sum()
        fn = (y[va] == 1).sum() - tp
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]:
            best = (f1, thr)
    print("val F1 %.3f at thr %.2f" % best)
    # финальное обучение на всех данных
    clf.fit((X - mu) / sd, y)
    d = dict(mu=mu, sd=sd)
    export_mlp(clf, os.path.join(a.out, "verifier.npz"), d)
    open(os.path.join(a.out, "verifier_threshold.txt"), "w").write("%.3f" % best[1])
    print("saved verifier + threshold", best[1])


if __name__ == "__main__":
    main()
