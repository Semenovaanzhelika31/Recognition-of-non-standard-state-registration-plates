#!/usr/bin/env python3
"""Обучение классификатора символов (22 символа ГОСТ + «не символ»).

Данные генерируются на лету из генератора знаков (seed фиксирован):
    python train/train_recognizer.py --n 20000 --seed 1 --out models/recognizer.npz
Окна берутся по слотам выпрямленного знака: положительные — малые сдвиги от
центра слота, класс none — окна, сдвинутые на соседний символ/фон, и окно
на месте несуществующей третьей цифры региона.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anpr import gost  # noqa: E402
from anpr.features import cut_windows, featurize  # noqa: E402
from train.synth_crops import sample_crop  # noqa: E402


def windows_for_crop(s, rng):
    gray, ptype, text, rl = s["gray"], s["ptype"], s["text"], s["region_len"]
    slots = gost.canon_slots(ptype, rl)
    patches, labels = [], []
    for ch, (kind, box) in zip(text, slots):
        bw, bh = box[2] - box[0], box[3] - box[1]
        c = gost.CHAR_TO_CLS[ch]
        sh = [(rng.uniform(-0.2, 0.2) * bw, rng.uniform(-0.12, 0.12) * bh) for _ in range(2)]
        patches.append(cut_windows(gray, box, sh)); labels += [c, c]
        sg = rng.choice([-1, 1])
        sh = [(sg * rng.uniform(0.55, 1.1) * bw, rng.uniform(-0.15, 0.15) * bh)]
        patches.append(cut_windows(gray, box, sh)); labels.append(gost.NONE_CLS)
    if rl == 2:   # на месте третьей цифры региона — «none»
        box = gost.canon_slots(ptype, 3)[-1][1]
        sh = [(rng.uniform(-0.2, 0.2) * (box[2] - box[0]), 0.0)]
        patches.append(cut_windows(gray, box, sh)); labels.append(gost.NONE_CLS)
    # случайное окно фона/мусора
    h, w = gray.shape
    box = slots[0][1]
    bw, bh = box[2] - box[0], box[3] - box[1]
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    patches.append(cut_windows(gray, (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), [(0, 0)]))
    # если окно случайно попало на символ, метка шумная; доля таких мала
    labels.append(gost.NONE_CLS)
    return np.concatenate(patches), np.array(labels)


def worker(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n):
        s = sample_crop(rng)
        p, l = windows_for_crop(s, rng)
        X.append(featurize(p).astype(np.float16)); Y.append(l)
    return np.concatenate(X), np.concatenate(Y)


def export_mlp(clf, path, extra=None):
    d = dict(n_layers=len(clf.coefs_))
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        d[f"W{i}"] = W.astype(np.float32); d[f"b{i}"] = b.astype(np.float32)
    if extra:
        d.update(extra)
    np.savez_compressed(path, **d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden", type=int, nargs="+", default=[320, 160])
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "recognizer.npz"))
    a = ap.parse_args()
    t0 = time.time()
    chunk = 250
    jobs = [(a.seed * 100000 + i, chunk) for i in range(a.n // chunk)]
    with Pool(a.workers) as pool:
        res = pool.map(worker, jobs)
    X = np.concatenate([r[0] for r in res]); Y = np.concatenate([r[1] for r in res])
    print("windows", X.shape, "none share %.2f" % (Y == gost.NONE_CLS).mean(), "t=%.0fs" % (time.time() - t0), flush=True)
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=tuple(a.hidden), batch_size=256, learning_rate_init=1e-3,
                        max_iter=a.epochs, early_stopping=True, validation_fraction=0.05, n_iter_no_change=4,
                        alpha=1e-5, random_state=a.seed, verbose=True)
    clf.fit(X.astype(np.float32), Y)
    print("val acc %.4f" % clf.best_validation_score_, "t=%.0fs" % (time.time() - t0))
    export_mlp(clf, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
