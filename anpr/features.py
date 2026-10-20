"""Окна символов и признаки (нормированные пиксели + HOG)."""
import cv2
import numpy as np

from .gost import NONE_CLS

PATCH_W, PATCH_H = 24, 32
N_FEAT = 3 * (PATCH_H // 2) * (PATCH_W // 2)   # пиксели + gx + gy, 16x12 каждый


def cut_windows(gray, box, shifts, pad_x=1.35, pad_y=1.18):
    """Вырезает окна 24x32 вокруг слота box=(x0,y0,x1,y1) со сдвигами shifts=[(dx,dy)].

    Окно линейным преобразованием приводится к 24x32, нормируется по 5–95 перцентилям.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) * pad_x / 2, (y1 - y0) * pad_y / 2
    sx, sy = PATCH_W / (2 * hw), PATCH_H / (2 * hh)
    out = np.empty((len(shifts), PATCH_H, PATCH_W), np.uint8)
    for i, (dx, dy) in enumerate(shifts):
        M = np.array([[sx, 0, PATCH_W / 2 - sx * (cx + dx)], [0, sy, PATCH_H / 2 - sy * (cy + dy)]], np.float32)
        out[i] = cv2.warpAffine(gray, M, (PATCH_W, PATCH_H), flags=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    return out


def featurize(patches):
    """patches: (N,32,24) uint8 -> (N,N_FEAT) float32.

    Признаки: нормированные пиксели 16x12 и компоненты градиента gx, gy 16x12
    (усреднение 2x2). Все операции плотные и быстрые на CPU.
    """
    n = len(patches)
    flat = patches.reshape(n, -1).astype(np.float32)
    srt = np.sort(flat, axis=1)
    lo = srt[:, int(0.05 * (flat.shape[1] - 1))][:, None]
    hi = srt[:, int(0.95 * (flat.shape[1] - 1))][:, None]
    norm = np.clip((flat - lo) / np.maximum(hi - lo, 12.0), 0, 1).reshape(n, PATCH_H, PATCH_W)
    gx = np.zeros_like(norm); gy = np.zeros_like(norm)
    gx[:, :, 1:-1] = norm[:, :, 2:] - norm[:, :, :-2]
    gy[:, 1:-1, :] = norm[:, 2:, :] - norm[:, :-2, :]
    ds = lambda a: (0.25 * (a[:, 0::2, 0::2] + a[:, 1::2, 0::2] + a[:, 0::2, 1::2] + a[:, 1::2, 1::2])).reshape(n, -1)
    return np.concatenate([ds(norm), 2.0 * ds(gx), 2.0 * ds(gy)], 1)
