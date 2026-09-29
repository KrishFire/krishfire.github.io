"""sharpen.py: Part 2.1, unsharp masking."""

import numpy as np
from scipy.signal import convolve2d
from utils import load, save, stretch, blur, gaussian_2d


def unsharp_filter(sigma, alpha):
    """Build the whole sharpening step as one filter.

    sharpened = image + alpha * (image - blurred)
              = (1 + alpha) * image - alpha * blurred

    "image" is the same as convolving with an impulse (a single 1 in the middle),
    so the filter is (1 + alpha) * impulse - alpha * gaussian.
    """
    G = gaussian_2d(sigma)
    impulse = np.zeros_like(G)
    impulse[G.shape[0] // 2, G.shape[1] // 2] = 1
    return (1 + alpha) * impulse - alpha * G


def sharpen(im, sigma, alpha):
    """Apply the unsharp filter to each colour channel."""
    k = unsharp_filter(sigma, alpha)
    return np.dstack([convolve2d(im[:, :, c], k, mode="same", boundary="symm")
                      for c in range(3)])


def show_steps(name, sigma, alphas):
    im = load(f"data/{name}")
    stem = name.split(".")[0]
    low = blur(im, sigma)
    high = im - low
    save(low, f"out/2_1_{stem}_blurred.png")
    save(stretch(high), f"out/2_1_{stem}_high.png")
    for a in alphas:
        save(sharpen(im, sigma, a), f"out/2_1_{stem}_alpha{a}.png")


def blur_then_sharpen(name, sigma, alpha):
    """Start from a sharp photo, blur it on purpose, then try to get it back."""
    im = load(f"data/{name}")
    stem = name.split(".")[0]
    blurry = blur(im, sigma)
    fixed = np.clip(sharpen(blurry, sigma, alpha), 0, 1)
    save(blurry, f"out/2_1_{stem}_blurred_on_purpose.png")
    save(fixed, f"out/2_1_{stem}_resharpened.png")
    print(f"{stem}: average error vs original  blurred {np.abs(blurry - im).mean():.4f}"
          f"   resharpened {np.abs(fixed - im).mean():.4f}")


def main():
    show_steps("taj.jpg", sigma=2, alphas=[0.5, 1, 2, 4])
    show_steps("harvesters.jpg", sigma=2, alphas=[0.5, 1, 2, 4])   # a soft 1900s photo from project 1
    blur_then_sharpen("building.jpg", sigma=2, alpha=1.5)           # a sharp photo from project 0


if __name__ == "__main__":
    main()
