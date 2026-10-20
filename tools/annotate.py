#!/usr/bin/env python3
"""Ручная разметка СОБСТВЕННЫХ реальных фото для датасета (нужен экран; запускать на своём ПК).

    python tools/annotate.py --src my_photos --root dataset --source own_photo --license own_photo \
        --conditions day,angle

Управление (окно OpenCV):
  ЛКМ x4        — углы знака по часовой стрелке от левого верхнего (TL, TR, BR, BL)
  ПКМ / z       — отменить последнюю точку;  c — стереть все точки текущего знака
  Enter         — завершить знак: в консоли ввести номер (A123BC77, '#' = нечитаемая позиция)
  1/2/3/4       — тип знака: 1=type1, 2=type1a, 3=type1b, 4=other
  v             — переключить is_vehicle (по умолчанию 1: знак на транспортном средстве)
  b             — режим размытия: протянуть прямоугольник мышью (лица людей обязательно закрыть!)
  n             — следующее изображение (сохранить);  s — пропустить изображение;  q — выход
Изображения копируются в dataset/images/real/, метка — в labels/<имя>.txt, строка — в meta.csv.
Номер вводится ВРУЧНУЮ человеком: автоматическая разметка чужой моделью запрещена ТЗ.
"""
import argparse, csv, os, re, shutil, sys
import cv2
import numpy as np

LET = "ABEKMHOPCTYX"
RE_NUM = re.compile(r"^[%s#][0-9#]{3}[%s#]{2}[0-9#]{2,3}$" % (LET, LET))
CLS = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}
KEYS = {ord("1"): "type1", ord("2"): "type1a", ord("3"): "type1b", ord("4"): "other"}
FIELDS = ["image", "plate_num", "plate_type", "bbox", "quad", "is_vehicle", "is_synthetic", "source", "license",
          "conditions"]


def write_annotation(root, src_path, plates, source, license_, conditions, img=None):
    """plates: [dict(quad 4x2, plate_num, plate_type, is_vehicle)]. Возвращает имя файла в датасете."""
    os.makedirs(os.path.join(root, "images", "real"), exist_ok=True)
    os.makedirs(os.path.join(root, "labels"), exist_ok=True)
    base = os.path.basename(src_path)
    name = "real_" + re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    dst = os.path.join(root, "images", "real", name)
    if img is not None:
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        shutil.copyfile(src_path, dst)
    im = cv2.imread(dst); H, W = im.shape[:2]
    lab = []
    meta = os.path.join(root, "meta.csv")
    new = not os.path.isfile(meta)
    with open(meta, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter=";")
        if new:
            w.writeheader()
        for p in plates:
            q = np.asarray(p["quad"], float)
            x0, y0 = q.min(0); x1, y1 = q.max(0)
            lab.append("%d %.6f %.6f %.6f %.6f " % (CLS[p["plate_type"]], (x0 + x1) / 2 / W, (y0 + y1) / 2 / H,
                                                  (x1 - x0) / W, (y1 - y0) / H)
                       + " ".join("%.6f" % v for v in (q / [W, H]).reshape(-1)))
            w.writerow(dict(image=f"images/real/{name}", plate_num=p["plate_num"] if p["plate_type"] != "other" else "#",
                            plate_type=p["plate_type"], bbox="%d,%d,%d,%d" % (x0, y0, x1 - x0, y1 - y0),
                            quad=",".join("%d" % round(v) for v in q.reshape(-1)), is_vehicle=int(p["is_vehicle"]),
                            is_synthetic=0, source=source, license=license_, conditions=conditions))
    open(os.path.join(root, "labels", os.path.splitext(name)[0] + ".txt"), "w").write("\n".join(lab) + ("\n" if lab else ""))
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--source", default="own_photo")
    ap.add_argument("--license", default="own_photo")
    ap.add_argument("--conditions", default="day")
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.src) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    done = set()
    meta = os.path.join(a.root, "meta.csv")
    if os.path.isfile(meta):
        done = {os.path.basename(r["image"]) for r in csv.DictReader(open(meta, encoding="utf-8"), delimiter=";")}
    for fn in files:
        if "real_" + fn in done:
            continue
        path = os.path.join(a.src, fn)
        img = cv2.imread(path)
        if img is None:
            continue
        H, W = img.shape[:2]
        sc = min(1.0, 1400 / max(H, W))
        plates, pts, ptype, veh = [], [], "type1", 1
        blur_mode, drag = False, None

        def on_mouse(ev, x, y, fl, _):
            nonlocal pts, drag
            if blur_mode:
                if ev == cv2.EVENT_LBUTTONDOWN:
                    drag = (x, y)
                elif ev == cv2.EVENT_LBUTTONUP and drag:
                    x0, y0 = int(min(drag[0], x) / sc), int(min(drag[1], y) / sc)
                    x1, y1 = int(max(drag[0], x) / sc), int(max(drag[1], y) / sc)
                    if x1 - x0 > 3 and y1 - y0 > 3:
                        roi = img[y0:y1, x0:x1]
                        k = max(3, (min(roi.shape[:2]) // 2) | 1)
                        img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
                    drag = None
            else:
                if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                    pts.append((x / sc, y / sc))
                elif ev == cv2.EVENT_RBUTTONDOWN and pts:
                    pts.pop()

        cv2.namedWindow("annotate"); cv2.setMouseCallback("annotate", on_mouse)
        while True:
            vis = cv2.resize(img, None, fx=sc, fy=sc) if sc < 1 else img.copy()
            for p in plates:
                cv2.polylines(vis, [(np.array(p["quad"]) * sc).astype(np.int32)], True, (0, 255, 0), 2)
            for i, p in enumerate(pts):
                cv2.circle(vis, (int(p[0] * sc), int(p[1] * sc)), 4, (0, 0, 255), -1)
            cv2.putText(vis, f"{fn} type={ptype} vehicle={veh} blur={blur_mode} plates={len(plates)}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.imshow("annotate", vis)
            k = cv2.waitKey(30) & 0xFF
            if k in KEYS: ptype = KEYS[k]
            elif k == ord("v"): veh = 1 - veh
            elif k == ord("b"): blur_mode = not blur_mode
            elif k == ord("z") and pts: pts.pop()
            elif k == ord("c"): pts = []
            elif k in (13, 10) and len(pts) == 4:
                num = "#"
                if ptype != "other":
                    while True:
                        num = input(f"[{fn}] номер знака (A123BC77; # — нечитаемо): ").strip().upper()
                        if RE_NUM.match(num):
                            break
                        print("неверный формат; буквы только из", LET)
                plates.append(dict(quad=pts, plate_num=num, plate_type=ptype, is_vehicle=veh)); pts = []
            elif k == ord("n"):
                write_annotation(a.root, path, plates, a.source, a.license, a.conditions, img=img); break
            elif k == ord("s"): break
            elif k == ord("q"): cv2.destroyAllWindows(); return
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
