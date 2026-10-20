#!/usr/bin/env python3
"""Сопоставление результата run.py с эталоном (meta.csv) на подмножестве изображений.

Знак считается найденным, если совпали image, plate_num и plate_type. Знаки с is_vehicle=0 не штрафуются:
верное предсказание для них игнорируется (ни TP, ни FP), пропуск не считается FN. Знаки типа other
не входят в эталон целевых: любое предсказание на изображении сверх совпавших — FP.

    python tools/evaluate.py --pred result.csv --gt dataset/meta.csv --images-dir eval_images
"""
import argparse, collections, csv, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True); ap.add_argument("--gt", required=True); ap.add_argument("--images-dir", required=True)
a = ap.parse_args()
names = {f for f in os.listdir(a.images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))}
gt = collections.defaultdict(list)
for r in csv.DictReader(open(a.gt, encoding="utf-8"), delimiter=";"):
    n = os.path.basename(r["image"])
    if n in names:
        q = np.array(list(map(float, r["quad"].split(","))), float).reshape(4, 2)
        r["w"] = float(np.linalg.norm(q[1] - q[0])); gt[n].append(r)
pred = collections.defaultdict(list)
for r in csv.DictReader(open(a.pred, encoding="utf-8"), delimiter=";"):
    pred[r["image"]].append(r)
tp = fp = fn = ign = 0
by_c = collections.defaultdict(lambda: [0, 0])
by = collections.defaultdict(lambda: [0, 0]); by_t = collections.defaultdict(lambda: [0, 0]); n_empty_fp = 0
conf_tp, conf_fp = [], []
for n in names:
    G = [g for g in gt[n] if g["plate_type"] != "other"]
    used = set()
    for p in pred[n]:
        hit = None
        for i, g in enumerate(G):
            if i not in used and g["plate_num"] == p["plate_num"] and g["plate_type"] == p["plate_type"]:
                hit = i; break
        if hit is None:
            fp += 1; conf_fp.append(float(p["confidence"]))
            if not G: n_empty_fp += 1
        else:
            used.add(hit)
            if G[hit]["is_vehicle"] == "0": ign += 1
            else: tp += 1; conf_tp.append(float(p["confidence"]))
    for i, g in enumerate(G):
        if g["is_vehicle"] == "0": continue
        b = "<60px" if g["w"] < 60 else "60-120" if g["w"] < 120 else ">=120"
        by[b][1] += 1; by_t[g["plate_type"]][1] += 1
        for c in g["conditions"].split(","):
            if g["w"] >= 120: by_c[c][1] += 1; by_c[c][0] += (i in used)
        if i in used: by[b][0] += 1; by_t[g["plate_type"]][0] += 1
        else: fn += 1
P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); F = 2 * P * R / max(P + R, 1e-9)
print(f"изображений {len(names)}  TP {tp}  FP {fp}  FN {fn}  (проигнорировано знаков на щитах: {ign})")
print(f"precision {P:.3f}  recall {R:.3f}  F1 {F:.3f}  (точное совпадение номера и типа)")
print("recall по размеру знака:", {k: f"{v[0]}/{v[1]}={v[0]/max(v[1],1):.2f}" for k, v in sorted(by.items())})
print("recall по условиям (только знаки >=120 px):", {k: f"{v[0]}/{v[1]}={v[0]/max(v[1],1):.2f}" for k, v in sorted(by_c.items())})
print("recall по типу:", {k: f"{v[0]}/{v[1]}={v[0]/max(v[1],1):.2f}" for k, v in sorted(by_t.items())})
if conf_tp: print("confidence TP: median %.2f; FP: median %.2f" % (np.median(conf_tp), np.median(conf_fp) if conf_fp else float('nan')))
