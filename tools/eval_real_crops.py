#!/usr/bin/env python3
"""Проверка на реальных вырезках внешнего набора Kaggle «Russian License Plate Characters - 23 Classes»
(CC BY 4.0, dimahkiin). Разметка этого набора сделана автором моделью, поэтому цифры — согласие с этой
разметкой, а не точность относительно полностью ручной.

Считает на отложенных (по тексту номера) вырезках:
  geom  — точное чтение при известной геометрии знака (качество самого распознавателя);
  e2e   — полный пайплайн (поиск знака + чтение + порог) на вырезке с полями, увеличенной в --scale раз.

    python tools/eval_real_crops.py --images D:\\kaggle\\images\\val --labels D:\\kaggle\\labels\\val --scale 3
"""
import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anpr.pipeline import PlateReader  # noqa: E402
from anpr.recognize import decode  # noqa: E402
from train import real_crops as rc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()
    _, test = rc.split_by_text(rc.list_samples(a.images, a.labels))
    test = test[:a.limit]
    R = PlateReader(a.models)
    geom = em = ex = 0
    for ip, t, b in test:
        im = cv2.imread(ip)
        C = rc.fit_canon(im, t, b)
        v = max(decode(C, "type1", R.rec, np.array([-0.14, -0.07, 0, 0.07, 0.14]), np.array([-0.06, 0, 0.06])),
                key=lambda x: x["total"])
        geom += v["text"] == t
        big = cv2.resize(im, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_CUBIC)
        pad = int(0.6 * big.shape[0])
        big = cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        pr = R.read(big)
        em += bool(pr)
        ex += any(p["plate_num"] == t for p in pr)
    n = len(test)
    print("отложенных вырезок: %d" % n)
    print("geom: точное чтение при известной геометрии %.3f" % (geom / n))
    print("e2e (масштаб x%g): выдан знак %.3f, номер точно верный %.3f" % (a.scale, em / n, ex / n))


if __name__ == "__main__":
    main()
