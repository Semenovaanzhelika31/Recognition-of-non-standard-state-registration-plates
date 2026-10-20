#!/usr/bin/env python3
"""Диагностика потерь: для каждого эталонного знака — найден ли кандидат, верен ли текст, прошёл ли порог."""
import argparse, collections, csv, os, pickle, sys
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anpr.pipeline import PlateReader, poly_iou
from anpr.recognize import is_yellow

ap = argparse.ArgumentParser()
ap.add_argument("--images-dir", required=True); ap.add_argument("--gt", default=os.path.join(ROOT, "dataset", "meta.csv"))
ap.add_argument("--cache", default=None); ap.add_argument("--min-w", type=float, default=0)
a = ap.parse_args()
R = PlateReader()
names = sorted(f for f in os.listdir(a.images_dir) if f.endswith(".jpg"))
if a.cache and os.path.exists(a.cache):
    cache = pickle.load(open(a.cache, "rb"))
else:
    cache = {}
    for n in names:
        res = R.analyze(cv2.imread(os.path.join(a.images_dir, n)), prune=-9.0, max_cands=20)
        cache[n] = [dict(text=r["text"], ptype=r["ptype"], yellow=bool(is_yellow(r["stats"])), quad=r["quad_img"], feat=r["feat"],
                         score=R.score(r), total=r["total"], mean_lp=r["mean_lp"]) for r in res]
    if a.cache: pickle.dump(cache, open(a.cache, "wb"))
gt = collections.defaultdict(list)
for r in csv.DictReader(open(a.gt, encoding="utf-8"), delimiter=";"):
    n = os.path.basename(r["image"])
    if n in cache and r["plate_type"] != "other" and r["is_vehicle"] == "1":
        q = np.array(list(map(float, r["quad"].split(","))), np.float32).reshape(4, 2)
        r["q"] = q; r["w"] = float(np.linalg.norm(q[1] - q[0])); gt[n].append(r)
cat = collections.defaultdict(collections.Counter)
for n, G in gt.items():
    for g in G:
        b = "<60" if g["w"] < 60 else "60-120" if g["w"] < 120 else ">=120"
        cs = [(poly_iou(g["q"], c["quad"]), c) for c in cache[n]]
        cs = [x for x in cs if x[0] > 0.4]
        if not cs: cat[b]["1 нет кандидата (IoU>0.4)"] += 1; continue
        ok = [c for _, c in cs if c["text"] == g["plate_num"]]
        if not ok:
            best = max(cs, key=lambda x: x[0])[1]
            err = sum(x != y for x, y in zip(best["text"], g["plate_num"])) + abs(len(best["text"]) - len(g["plate_num"]))
            cat[b]["2 текст неверен (>=3 ош.)" if err >= 3 else "2 текст неверен (1-2 ош.)"] += 1; continue
        c = max(ok, key=lambda c: c["score"])
        t = "type1a" if c["ptype"] == "type1a" else ("type1b" if c["yellow"] else "type1")
        if t != g["plate_type"]: cat[b]["3 верный текст, неверный тип"] += 1
        else: cat[b]["4 верно (score=%s)" % ("≥0.56" if c["score"] >= 0.56 else "<0.56")] += 1
for b in ("<60", "60-120", ">=120"):
    tot = sum(cat[b].values()); print(b, "всего", tot)
    for k, v in sorted(cat[b].items()): print("   %-34s %4d  %.2f" % (k, v, v / tot))
