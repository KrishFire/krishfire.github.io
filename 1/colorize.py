import os 
import time
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def load_plate(path):
    im = np.asarray(Image.open(path))
    if im.dtype == np.uint8:
        im = im / 255.0
    elif im.dtype == np.uint16:
        im = im / 65535.0
    return im.astype(np.float32)


def split_channels(plate):
    h = plate.shape[0] // 3
    return plate[0:h], plate[h:2*h], plate[2*h:3*h]


def inner(im, keep=0.8):
    h, w = im.shape
    dh = int(h * (1 - keep) / 2)
    dw = int(w * (1 - keep) / 2)
    return im[dh:h - dh, dw:w - dw]


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a*a).sum()) * np.sqrt((b*b).sum())
    if denom == 0:
        return 0.0
    return float((a*b).sum() / denom)


def l2(a, b): 
    d = a - b
    return float(np.sqrt((d*d).sum()))


def score(a, b, metric):
    if metric == "ncc":
        return ncc(a, b)
    if metric == "l2":
        return -l2(a,b)
    raise ValueError("metric must be 'ncc' or 'l2'")


def best_shift(moving, fixed_inner, center, radius, metric, keep):
    cy, cx = center
    best_score = -np.inf
    best = center
    for dy in range(cy - radius, cy + radius + 1):
        for dx in range(cx - radius, cx + radius + 1):
            shifted = np.roll(moving, (dy, dx), axis=(0, 1))
            s = score(inner(shifted, keep), fixed_inner, metric)
            if s > best_score:
                best_score = s
                best = (dy, dx)
    return best


def align_single(moving, fixed, radius=15, metric="ncc", keep=0.8):
    return best_shift(moving, inner(fixed, keep), (0, 0), radius, metric, keep)


def downsample(im):
    h, w = im.shape
    h -= h % 2          
    w -= w % 2
    im = im[:h, :w]
    return (im[0::2, 0::2] + im[1::2, 0::2] +
            im[0::2, 1::2] + im[1::2, 1::2]) / 4.0


def align_pyramid(moving, fixed, metric="ncc", keep=0.8,
                  min_size=300, coarse_radius=15, fine_radius=2):
    if max(moving.shape) <= min_size:
        return align_single(moving, fixed, coarse_radius, metric, keep)

    dy, dx = align_pyramid(downsample(moving), downsample(fixed),metric, keep, min_size, coarse_radius, fine_radius)

    return best_shift(moving, inner(fixed, keep), (dy * 2, dx * 2), fine_radius, metric, keep)


def colorize(path, metric="ncc", single_scale=False):
    b, g, r = split_channels(load_plate(path))
    align = align_single if single_scale else align_pyramid
    g_off = align(g, b, metric=metric)
    r_off = align(r, b, metric=metric)
    g_aligned = np.roll(g, g_off, axis=(0, 1))
    r_aligned = np.roll(r, r_off, axis=(0, 1))
    return np.dstack([r_aligned, g_aligned, b]), g_off, r_off


def save_jpg(im, path, max_width=1000):
    pic = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
    if pic.width > max_width:
        pic = pic.resize((max_width, round(pic.height * max_width / pic.width)), Image.LANCZOS)
    pic.save(path, quality=92)



def edges(im):
    gy = np.zeros_like(im)
    gx = np.zeros_like(im)
    gy[1:-1, :] = im[2:, :] - im[:-2, :]
    gx[:, 1:-1] = im[:, 2:] - im[:, :-2]
    return np.sqrt(gx * gx + gy * gy)


def colorize_edges(path, metric="ncc"):
    b, g, r = split_channels(load_plate(path))
    g_off = align_pyramid(edges(g), edges(b), metric=metric)
    r_off = align_pyramid(edges(r), edges(b), metric=metric)
    return (np.dstack([np.roll(r, r_off, axis=(0, 1)), np.roll(g, g_off, axis=(0, 1)), b]), g_off, r_off)


DATA_DIR = "CS180_fa2026_proj1_data"
OUT_DIR = "out"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(DATA_DIR) if f.endswith((".jpg", ".tif")))

    print(f"{'image':24} {'G (dy,dx)':>14} {'R (dy,dx)':>14} {'time':>7}")
    print("-" * 62)
    for name in names:
        path = os.path.join(DATA_DIR, name)
        start = time.time()
        color, g_off, r_off = colorize(path)
        elapsed = time.time() - start
        stem = name.rsplit(".", 1)[0]
        save_jpg(color, os.path.join(OUT_DIR, stem + ".jpg"))
        print(f"{stem:24} {str(g_off):>14} {str(r_off):>14} {elapsed:6.1f}s")


if __name__ == "__main__":
    main()



