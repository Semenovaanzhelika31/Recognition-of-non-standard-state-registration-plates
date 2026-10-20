#!/usr/bin/env python3
"""Проверка распознавателя на синтетических выпрямленных кропах (held-out seed)."""
import argparse, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anpr.recognize import MLP, decode
from train.synth_crops import sample_crop

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "models", "recognizer.npz"))
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--jit", type=float, default=None)
a = ap.parse_args()
rec = MLP(a.model)
rng = np.random.default_rng(a.seed)
dx = np.array([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30])
rows = []
t = time.time()
for _ in range(a.n):
    s = sample_crop(rng, jit=a.jit)
    v = decode(s["gray"], s["ptype"], rec, dx, np.array([0.0]))[0]
    rows.append((s["wpx"], s["text"], v["text"], s["region_len"] == v["region_len"]))
print("ms/decode", (time.time() - t) / a.n * 1000)
w = np.array([r[0] for r in rows])
for lo, hi in [(0, 60), (60, 100), (100, 160), (160, 260), (260, 1e9)]:
    m = [r for r in rows if lo <= r[0] < hi]
    if not m:
        continue
    em = np.mean([r[1] == r[2] for r in m])
    ca = np.mean([np.mean([x == y for x, y in zip(r[1], r[2])]) if len(r[1]) == len(r[2]) else 0 for r in m])
    print(f"width {lo:>4}-{hi:<6.0f} n={len(m):3d} exact={em:.3f} char_acc={ca:.3f}")
print("overall exact %.3f" % np.mean([r[1] == r[2] for r in rows]))
