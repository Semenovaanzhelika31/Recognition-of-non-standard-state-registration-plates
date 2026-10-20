#!/usr/bin/env python3
"""Полнота и качество геометрии кандидатов детектора на синтетическом наборе.

    python tools/bench_detect.py --root EVSYN --n 400

Для каждого целевого знака (type1/1a/1b на транспортном средстве) берётся лучший кандидат по IoU
четырёхугольников; считаются доли IoU>0.3/0.5/0.7 по ширине знака и по типу, число кандидатов на кадр и время.
Опции detect.candidates можно передавать как --opt key=value (например --opt use_edge=1).
"""
import argparse
import collections
import csv
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anpr import detect  # noqa: E402
from anpr.pipeline import poly_iou  # noqa: E402

BUCKETS = [("<40", 0, 40), ("40-60", 40, 60), ("60-90", 60, 90), ("90-120", 90, 120), (">=120", 120, 1e9)]


def bucket(w):
    for name, lo, hi in BUCKETS:
        if lo <= w < hi:
            return name


def parse_opts(items):
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k] = float(v) if "." in v else int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--opt", action="append")
    a = ap.parse_args()
    opts = parse_opts(a.opt)
    rows = list(csv.DictReader(open(os.path.join(a.root, "meta.csv"), encoding="utf-8"), delimiter=";"))
    by = collections.OrderedDict()
    for r in rows:
        by.setdefault(r["image"], []).append(r)
    names = list(by)[:a.n]
    best = collections.defaultdict(list)
    ncand, tt = [], []
    for im in names:
        g = cv2.cvtColor(cv2.imread(os.path.join(a.root, im)), cv2.COLOR_BGR2GRAY)
        t = time.time()
        cands = detect.candidates(g, **opts)
        tt.append(time.time() - t)
        ncand.append(len(cands))
        for r in by[im]:
            if r["plate_type"] == "other" or r["is_vehicle"] != "1":
                continue
            q = np.array(list(map(float, r["quad"].split(","))), np.float32).reshape(4, 2)
            iou = max([poly_iou(q, c["quad"]) for c in cands] + [0.0])
            w = float(np.linalg.norm(q[1] - q[0]))
            best[("все", bucket(w))].append(iou)
            best[(r["plate_type"], bucket(w))].append(iou)
    print("кадров %d, кандидатов/кадр %.1f, время детекции %.0f мс/кадр" % (len(names), np.mean(ncand), 1000 * np.mean(tt)))
    print("%-8s %-7s %4s  %6s %6s %6s %6s" % ("тип", "ширина", "n", ">0.3", ">0.5", ">0.7", "mean"))
    for typ in ("все", "type1", "type1a", "type1b"):
        for name, _, _ in BUCKETS:
            v = np.array(best.get((typ, name), []))
            if len(v):
                print("%-8s %-7s %4d  %6.2f %6.2f %6.2f %6.2f" % (typ, name, len(v), (v > .3).mean(), (v > .5).mean(), (v > .7).mean(), v.mean()))


if __name__ == "__main__":
    main()
