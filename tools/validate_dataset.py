#!/usr/bin/env python3
"""Собственная проверка датасета (аналог validate_dataset.py организаторов, которого у нас нет).

Проверяет структуру каталогов, поля meta.csv, форматы номеров, соответствие изображений,
меток .txt и строк meta.csv, лицензии и источники, пересечение с отладочным набором (по хэшам),
сводную статистику относительно рекомендованных объёмов. Пишет отчёт в Markdown.

    python tools/validate_dataset.py --root dataset --report dataset/validation_report.md
    python tools/validate_dataset.py --root dataset --debug-dir /path/to/aisgorod_debug   # проверка на утечку
"""
import argparse, csv, hashlib, os, re, sys, collections

FIELDS = ["image", "plate_num", "plate_type", "bbox", "quad", "is_vehicle", "is_synthetic", "source", "license",
          "conditions"]
TYPES = {"type1", "type1a", "type1b", "other"}
COND = {"day", "night", "rain", "snow", "dirt", "glare", "motion_blur", "angle"}
LET = "ABEKMHOPCTYX"
RE_NUM = re.compile(r"^[%s#][0-9#]{3}[%s#]{2}[0-9#]{2,3}$" % (LET, LET))
SYN_LIC_OK = ("CC-BY-4.0", "CC BY 4.0", "own_photo", "CC0", "public domain")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--debug-dir", default=None)
    a = ap.parse_args()
    errs, warns = [], []
    root = a.root
    for d in ("images/real", "images/synthetic", "labels", "generator"):
        if not os.path.isdir(os.path.join(root, d)):
            errs.append(f"нет каталога {d}")
    for f in ("README.md", "LICENSE"):
        if not os.path.isfile(os.path.join(root, f)):
            errs.append(f"нет файла {f}")
    meta = a.meta or os.path.join(root, "meta.csv")
    if not os.path.isfile(meta):
        errs.append("нет meta.csv"); rows = []
    else:
        with open(meta, encoding="utf-8", newline="") as fh:
            rd = csv.DictReader(fh, delimiter=";")
            if rd.fieldnames != FIELDS:
                errs.append(f"заголовок meta.csv: {rd.fieldnames} != {FIELDS}")
            rows = list(rd)
    by_img = collections.defaultdict(list)
    stats = collections.Counter(); uniq = collections.defaultdict(set); imgs_by = collections.defaultdict(set)
    for i, r in enumerate(rows, 2):
        tag = f"meta.csv:{i}"
        by_img[r["image"]].append(r)
        if not os.path.isfile(os.path.join(root, r["image"])):
            errs.append(f"{tag}: нет файла {r['image']}")
        if r["plate_type"] not in TYPES:
            errs.append(f"{tag}: plate_type={r['plate_type']}")
        if r["plate_type"] != "other" and not RE_NUM.match(r["plate_num"]):
            errs.append(f"{tag}: формат номера {r['plate_num']!r}")
        try:
            b = list(map(float, r["bbox"].split(",")));  assert len(b) == 4 and b[2] > 0 and b[3] > 0
        except Exception:
            errs.append(f"{tag}: bbox {r['bbox']!r}")
        try:
            q = list(map(float, r["quad"].split(","))); assert len(q) == 8
        except Exception:
            errs.append(f"{tag}: quad {r['quad']!r}")
        if r["is_vehicle"] not in ("0", "1"): errs.append(f"{tag}: is_vehicle")
        if r["is_synthetic"] not in ("0", "1"): errs.append(f"{tag}: is_synthetic")
        if not r["source"].strip() or not r["license"].strip():
            errs.append(f"{tag}: пустые source/license")
        for c in filter(None, r["conditions"].split(",")):
            if c not in COND: errs.append(f"{tag}: условие {c!r}")
        if r["is_synthetic"] == "0":
            if r["image"].startswith("images/synthetic"): errs.append(f"{tag}: real в synthetic/")
            if r["license"].strip().lower() in ("", "unknown", "?"): errs.append(f"{tag}: лицензия не указана")
        key = ("syn" if r["is_synthetic"] == "1" else "real", r["plate_type"])
        stats[key] += 1
        imgs_by[key].add(r["image"])
        if r["plate_type"] != "other": uniq[key].add(r["plate_num"])
    # метки
    n_lab = 0
    for sub in ("real", "synthetic"):
        d = os.path.join(root, "images", sub)
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
            lab = os.path.join(root, "labels", os.path.splitext(f)[0] + ".txt")
            if not os.path.isfile(lab):
                errs.append(f"нет метки для {sub}/{f}"); continue
            n_lab += 1
            lines = [l for l in open(lab).read().splitlines() if l.strip()]
            n_meta = len(by_img.get(f"images/{sub}/{f}", []))
            if len(lines) != n_meta:
                errs.append(f"{sub}/{f}: строк в метке {len(lines)} != строк в meta {n_meta}")
            for l in lines:
                v = l.split()
                if len(v) != 13 or int(v[0]) not in (0, 1, 2, 3) or any(not (-0.01 <= float(x) <= 1.01) for x in v[1:]):
                    errs.append(f"{lab}: неверная строка {l!r}")
    for im in by_img:
        pass
    # утечка отладочного набора
    if a.debug_dir:
        dbg = {md5(os.path.join(a.debug_dir, f)) for f in os.listdir(a.debug_dir)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))}
        for sub in ("real", "synthetic"):
            d = os.path.join(root, "images", sub)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if md5(os.path.join(d, f)) in dbg:
                        errs.append(f"{sub}/{f} совпадает с изображением отладочного набора")
    # объёмы
    need = {("real", "type1a"): (150, 50), ("real", "type1b"): (300, 100), ("real", "other"): (50, 0)}
    infos = []
    n_real_total = sum(len(imgs_by[k]) for k in need)
    if n_real_total == 0:
        infos.append("реальных изображений нет: датасет полностью синтетический (ТЗ допускает «собрать и/или сгенерировать»); "
                     "рекомендованные объёмы реальной части (type1a 150/50, type1b 300/100, other 50) не набирались")
    for k, (ni, nu) in (need.items() if n_real_total else []):
        have_i, have_u = len(imgs_by[k]), len(uniq[k])
        if have_i < ni or have_u < nu:
            warns.append(f"реальные данные {k[1]}: изображений {have_i}/{ni}, уникальных номеров {have_u}/{nu} — меньше рекомендованного минимума")
    d_syn = os.path.join(root, "images", "synthetic")
    n_syn = len([f for f in os.listdir(d_syn) if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if os.path.isdir(d_syn) else 0
    if n_syn < 5000:
        warns.append(f"синтетических изображений {n_syn} < 5000")
    lines = ["# Отчёт проверки датасета", "", f"Корень: `{root}`; строк meta.csv: {len(rows)}; файлов меток: {n_lab}; синтетических изображений: {n_syn}", "",
             "| источник | тип | знаков | изображений | уникальных номеров |", "|---|---|---|---|---|"]
    for k in sorted(stats):
        lines.append(f"| {k[0]} | {k[1]} | {stats[k]} | {len(imgs_by[k])} | {len(uniq[k]) if k[1] != 'other' else '-'} |")
    lines += ["", f"**Ошибок: {len(errs)}**, предупреждений: {len(warns)}", ""]
    lines += [f"- ОШИБКА: {e}" for e in errs[:200]] + [f"- ПРЕДУПРЕЖДЕНИЕ: {w}" for w in warns]
    lines += [f"- ИНФО: {i}" for i in infos]
    txt = "\n".join(lines)
    print(txt)
    if a.report:
        open(a.report, "w", encoding="utf-8").write(txt + "\n")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
