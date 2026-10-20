#!/usr/bin/env python3

import argparse
import csv
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from anpr.pipeline import PlateReader  # noqa: E402

EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=os.environ.get("PLATES_IMAGES"), help="каталог с изображениями")
    ap.add_argument("--out", default=os.environ.get("PLATES_OUT", "result.csv"), help="выходной CSV")
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--threshold", type=float, default=None, help="порог уверенности (по умолчанию из models/)")
    ap.add_argument("--recursive", action="store_true")
    a = ap.parse_args()
    if not a.images or not os.path.isdir(a.images):
        ap.error("укажите существующий каталог через --images или PLATES_IMAGES")
    reader = PlateReader(a.models, conf_thr=a.threshold)
    files = []
    if a.recursive:
        for d, _, fs in os.walk(a.images):
            files += [os.path.join(d, f) for f in fs if f.endswith(EXT)]
    else:
        files = [os.path.join(a.images, f) for f in os.listdir(a.images) if f.endswith(EXT)]
    files.sort()
    t0 = time.time()
    n_rows = 0
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";", lineterminator="\n")
        w.writerow(["image", "plate_num", "plate_type", "confidence"])
        for p in files:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                print("не удалось прочитать", p, file=sys.stderr)
                continue
            try:
                dets = reader.read(img)
            except Exception as e:  # один сбойный кадр не должен ронять весь прогон
                print("ошибка на", p, repr(e), file=sys.stderr)
                continue
            for d in dets:
                w.writerow([os.path.basename(p), d["plate_num"], d["plate_type"], "%.2f" % d["confidence"]])
                n_rows += 1
    dt = time.time() - t0
    print(f"изображений: {len(files)}, знаков: {n_rows}, среднее время {1000 * dt / max(len(files), 1):.0f} мс/изобр.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
