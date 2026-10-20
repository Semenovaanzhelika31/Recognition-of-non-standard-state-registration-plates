"""Геометрия знаков в каноническом (выпрямленном) виде и общие константы."""
import os
import sys

_GEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "generator")
if _GEN not in sys.path:
    sys.path.insert(0, _GEN)

from gostplate import (ALPHABET, DIGITS, LETTERS, SIZE_MM, layout)  # noqa: E402,F401

CAN_SCALE = {"type1": 0.6, "type1b": 0.6, "type1a": 0.8}   # пикселей на мм в каноническом виде
NONE_CLS = len(ALPHABET)                                    # класс «не символ»
N_CLS = NONE_CLS + 1
CHAR_TO_CLS = {c: i for i, c in enumerate(ALPHABET)}
LETTER_IDX = [CHAR_TO_CLS[c] for c in LETTERS]
DIGIT_IDX = [CHAR_TO_CLS[c] for c in DIGITS]
TYPES = ("type1", "type1a", "type1b")


def canon_size(ptype):
    w, h = SIZE_MM[ptype]
    s = CAN_SCALE[ptype]
    return int(round(w * s)), int(round(h * s))


def canon_slots(ptype, region_len):
    """[(kind, (x0,y0,x1,y1) в пикселях канонического изображения)]"""
    s = CAN_SCALE[ptype]
    return [(k, tuple(v * s for v in b)) for k, b in layout(ptype, region_len)]


def canon_corners(ptype):
    w, h = canon_size(ptype)
    import numpy as np
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
