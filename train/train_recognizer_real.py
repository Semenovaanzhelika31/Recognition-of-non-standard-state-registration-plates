#!/usr/bin/env python3
"""Обучение распознавателя на смеси синтетики и реальных вырезок (внешний набор, CC BY 4.0).

    python train/train_recognizer_real.py --images D:\\kaggle\\images\\val --labels D:\\kaggle\\labels\\val ^
        --synth-n 14000 --reps 6 --out models/recognizer.npz

Реальные вырезки делятся по тексту номера: 20% номеров целиком уходят в тест и в обучении не участвуют.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anpr import gost  # noqa: E402
from anpr.features import featurize  # noqa: E402
from train import real_crops as rc  # noqa: E402
from train.train_recognizer import export_mlp, windows_for_crop, worker  # noqa: E402


def augment(gray, rng):
    g = gray.astype(np.float32)
    if rng.random() < 0.6:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.3, 1.3))
    g = (g - g.mean()) * rng.uniform(0.6, 1.3) + g.mean() * rng.uniform(0.75, 1.2)
    sig = rng.uniform(0, 5)
    if sig > 0.5:
        g = g + rng.normal(0, sig, g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


def real_windows(samples, reps, seed):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for ip, text, boxes in samples:
        img = cv2.imread(ip)
        if img is None:
            continue
        for r in range(reps):
            C = rc.fit_canon(img, text, boxes, jitter=0.0 if r == 0 else rng.uniform(0.3, 1.2), rng=rng)
            if C is None:
                continue
            if r > 0:
                C = augment(C, rng)
            p, l = windows_for_crop(dict(gray=C, ptype="type1", text=text, region_len=len(text) - 6), rng)
            X.append(featurize(p).astype(np.float16)); Y.append(l)
    return np.concatenate(X), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--synth-n", type=int, default=14000)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, nargs="+", default=[320, 160])
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "recognizer_real.npz"))
    a = ap.parse_args()
    t0 = time.time()
    samples = rc.list_samples(a.images, a.labels)
    train, test = rc.split_by_text(samples)
    print("реальных вырезок: всего %d, обучение %d, тест %d" % (len(samples), len(train), len(test)), flush=True)
    Xr, Yr = real_windows(train, a.reps, a.seed)
    print("real windows", Xr.shape, "t=%.0fs" % (time.time() - t0), flush=True)
    chunk = 250
    jobs = [(a.seed * 100000 + i, chunk) for i in range(a.synth_n // chunk)]
    with Pool(a.workers) as pool:
        res = pool.map(worker, jobs)
    Xs = np.concatenate([r[0] for r in res]); Ys = np.concatenate([r[1] for r in res])
    print("synthetic windows", Xs.shape, "t=%.0fs" % (time.time() - t0), flush=True)
    X = np.concatenate([Xr, Xs]).astype(np.float32); Y = np.concatenate([Yr, Ys])
    del Xr, Xs
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=tuple(a.hidden), batch_size=256, learning_rate_init=1e-3,
                        max_iter=a.epochs, early_stopping=True, validation_fraction=0.05, n_iter_no_change=4,
                        alpha=1e-5, random_state=a.seed, verbose=True)
    clf.fit(X, Y)
    print("val acc %.4f" % clf.best_validation_score_, "t=%.0fs" % (time.time() - t0), flush=True)
    export_mlp(clf, a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
