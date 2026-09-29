"""hybrid.py: Part 2.2, hybrid images."""

import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from utils import load, save, blur


# ---------------------------------------------------------------- alignment

def pick_points(im, title, cache):
    """Click two points on the image (for faces, the two eyes). Saved so you only click once."""
    if os.path.exists(cache):
        return np.load(cache)
    plt.imshow(im)
    plt.title(title)
    pts = np.array(plt.ginput(2, timeout=0))
    plt.close()
    np.save(cache, pts)
    return pts


def align(im1, pts1, im2, pts2):
    """Move, turn and resize im2 so its two points land on im1's two points.

    The line between the two points tells us everything:
      its length gives the scale, its direction gives the rotation,
      and after scaling and turning, a shift puts the first point in place.
    """
    v1 = pts1[1] - pts1[0]
    v2 = pts2[1] - pts2[0]
    scale = np.linalg.norm(v1) / np.linalg.norm(v2)
    turn = np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])

    c, s = scale * np.cos(turn), scale * np.sin(turn)
    rot = np.array([[c, -s], [s, c]])
    shift = pts1[0] - rot @ pts2[0]
    M = np.hstack([rot, shift[:, None]])      # 2x3 matrix that cv2 understands

    h, w = im1.shape[:2]
    warped = cv2.warpAffine(im2, M, (w, h), flags=cv2.INTER_LINEAR)

    # warp a sheet of ones the same way to see which pixels got real image data
    valid = cv2.warpAffine(np.ones(im2.shape[:2]), M, (w, h)) > 0.999
    top, bottom, left, right = crop_to_valid(valid)
    return im1[top:bottom, left:right], warped[top:bottom, left:right], (top, left)


def crop_to_valid(valid):
    """Find a rectangle that only contains real pixels.

    Start from the box around everything valid, then keep trimming whichever
    edge has the most empty pixels until there are none left.
    """
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    while not valid[top:bottom, left:right].all():
        edges = [valid[top, left:right].mean(), valid[bottom - 1, left:right].mean(),
                 valid[top:bottom, left].mean(), valid[top:bottom, right - 1].mean()]
        worst = int(np.argmin(edges))
        if worst == 0:
            top += 1
        elif worst == 1:
            bottom -= 1
        elif worst == 2:
            left += 1
        else:
            right -= 1
    return top, bottom, left, right


# ---------------------------------------------------------------- the hybrid itself

def low_pass(im, sigma):
    return blur(im, sigma)


def high_pass(im, sigma):
    return im - blur(im, sigma)


def make_hybrid(far, near, sigma_low, sigma_high):
    """far: the image you should see from across the room (keeps its low frequencies).
    near: the image you should see up close (keeps its high frequencies)."""
    low = low_pass(far, sigma_low)
    high = high_pass(near, sigma_high)
    return np.clip(low + high, 0, 1), low, high


# ---------------------------------------------------------------- frequency pictures

def log_fft(im):
    """Log magnitude of the 2D Fourier transform, centred so low frequencies are in the middle."""
    gray = im.mean(axis=2) if im.ndim == 3 else im
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)


def save_fft_panel(images, labels, path):
    """All five panels share one brightness scale, taken from the first input,
    so you can actually see which frequencies each filter threw away."""
    ffts = [log_fft(im) for im in images]
    lo, hi = np.percentile(ffts[0], [1, 99.9])
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    for ax, f, label in zip(axes, ffts, labels):
        ax.imshow(f, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main

def to_gray(im):
    """Average the three channels, then copy the result back into all three."""
    g = im.mean(axis=2)
    return np.dstack([g, g, g])


def color_experiment(name, far, near, sigma_low, sigma_high):
    """Bells and whistles: which half of the hybrid should keep its colour?"""
    versions = {
        "gray": (to_gray(far), to_gray(near)),
        "color_far_only": (far, to_gray(near)),
        "color_near_only": (to_gray(far), near),
        "color_both": (far, near),
    }
    for label, (f, n) in versions.items():
        hybrid, _, _ = make_hybrid(f, n, sigma_low, sigma_high)
        save(hybrid, f"out/2_2_{name}_{label}.png")


def run(name, far_path, near_path, sigma_low, sigma_high, full_process=False, colors=False):
    far = load(far_path)
    near = load(near_path)
    p_far = pick_points(far, "far image: click 2 points", f"data/{name}_far_pts.npy")
    p_near = pick_points(near, "near image: click the same 2 points", f"data/{name}_near_pts.npy")
    far, near, _ = align(far, p_far, near, p_near)

    hybrid, low, high = make_hybrid(far, near, sigma_low, sigma_high)
    save(hybrid, f"out/2_2_{name}_hybrid.png")
    save(far, f"out/2_2_{name}_far_input.png")
    save(near, f"out/2_2_{name}_near_input.png")

    if colors:
        color_experiment(name, far, near, sigma_low, sigma_high)

    if full_process:
        save(low, f"out/2_2_{name}_low.png")
        save(np.clip(high + 0.5, 0, 1), f"out/2_2_{name}_high.png")   # +0.5 so negatives are visible
        save_fft_panel([far, near, low, high, hybrid],
                       ["far input", "near input", "low passed", "high passed", "hybrid"],
                       f"out/2_2_{name}_fft.png")


def main():
    # same roles as the starter code: Nutmeg is seen from far away, Derek up close
    run("derek_nutmeg", "data/hybrid/nutmeg.jpg", "data/hybrid/DerekPicture.jpg",
        sigma_low=8, sigma_high=4, full_process=True)

    # change over time: the Emir of Bukhara (1911) from far, me (2026) up close
    run("emir_me", "data/emir_face.jpg", "data/selfie.jpg",
        sigma_low=7, sigma_high=5, full_process=True, colors=True)

    # morph between two things: me from far, Nutmeg up close
    run("me_nutmeg", "data/selfie.jpg", "data/hybrid/nutmeg.jpg",
        sigma_low=8, sigma_high=4)


if __name__ == "__main__":
    main()
