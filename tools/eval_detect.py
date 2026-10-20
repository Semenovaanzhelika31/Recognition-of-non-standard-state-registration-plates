#!/usr/bin/env python3
"""Полнота генератора кандидатов на синтетическом наборе (по meta_synthetic.csv)."""
import argparse, os, sys, time, csv, collections
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anpr import detect

def poly_iou(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    ia, _ = cv2.intersectConvexConvex(a, b)
    ua = cv2.contourArea(a) + cv2.contourArea(b) - ia
    return ia / ua if ua > 0 else 0.0

ap = argparse.ArgumentParser()
ap.add_argument("--meta", default=os.path.join(ROOT, "dataset", "meta_synthetic.csv"))
ap.add_argument("--root", default=os.path.join(ROOT, "dataset"))
ap.add_argument("--start", type=int, default=4000)
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
rows = list(csv.DictReader(open(a.meta, encoding="utf-8"), delimiter=";"))
byimg = collections.OrderedDict()
for r in rows:
    byimg.setdefault(r["image"], []).append(r)
imgs = list(byimg)[a.start:a.start + a.n]
hit = collections.defaultdict(list); tcand = []; ncand = []
for im in imgs:
    g = cv2.cvtColor(cv2.imread(os.path.join(a.root, im)), cv2.COLOR_BGR2GRAY)
    t = time.time(); cands = detect.candidates(g); tcand.append(time.time() - t); ncand.append(len(cands))
    for r in byimg[im]:
        if r["plate_type"] == "other" or r["is_vehicle"] == "0":
            continue
        q = np.array(list(map(float, r["quad"].split(","))), np.float32).reshape(4, 2)
        wpx = np.linalg.norm(q[1] - q[0])
        best = max([poly_iou(q, c["quad"]) for c in cands] + [0.0])
        b = "<60" if wpx < 60 else "60-120" if wpx < 120 else ">120"
        hit[(r["plate_type"], b)].append(best)
print("candidates/img %.1f  time %.0f ms" % (np.mean(ncand), 1000 * np.mean(tcand)))
for k in sorted(hit):
    v = np.array(hit[k]); print(k, "n=%d IoU>0.3: %.2f  IoU>0.5: %.2f  mean %.2f" % (len(v), (v > .3).mean(), (v > .5).mean(), v.mean()))
