"""conv.py: Part 1.1, convolution from scratch using only numpy."""

import time
import numpy as np
from scipy.signal import convolve2d
from utils import load, save, stretch


def zero_pad(im, ph, pw):
    """Surround the image with ph rows of zeros on top and bottom, and pw columns on each side."""
    h, w = im.shape
    out = np.zeros((h + 2 * ph, w + 2 * pw))
    out[ph:ph + h, pw:pw + w] = im
    return out


def crop_same(full, im_shape, k_shape):
    """Cut the middle out of a "full" result so it is the same size as the input image.
    This matches the offset scipy uses, which also works for even sized kernels."""
    h, w = im_shape
    top = (k_shape[0] - 1) // 2
    left = (k_shape[1] - 1) // 2
    return full[top:top + h, left:left + w]


def conv_four_loops(im, k):
    """Convolution with four nested loops. Easy to read, very slow."""
    k = k[::-1, ::-1]                       # convolution flips the filter
    kh, kw = k.shape
    padded = zero_pad(im, kh - 1, kw - 1)   # "full" padding
    out_h = im.shape[0] + kh - 1
    out_w = im.shape[1] + kw - 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            total = 0.0
            for a in range(kh):
                for b in range(kw):
                    total += padded[i + a, j + b] * k[a, b]
            out[i, j] = total
    return crop_same(out, im.shape, (kh, kw))


def conv_two_loops(im, k):
    """Same thing, but the two inner loops become one numpy multiply and sum."""
    k = k[::-1, ::-1]
    kh, kw = k.shape
    padded = zero_pad(im, kh - 1, kw - 1)
    out_h = im.shape[0] + kh - 1
    out_w = im.shape[1] + kw - 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            window = padded[i:i + kh, j:j + kw]
            out[i, j] = (window * k).sum()
    return crop_same(out, im.shape, (kh, kw))


def main():
    im = load("data/selfie.jpg", gray=True, max_side=300)   # small, so the loop versions finish

    box = np.ones((9, 9)) / 81      # every pixel becomes the average of its 9x9 neighbourhood
    Dx = np.array([[1, -1]])
    Dy = np.array([[1], [-1]])

    for name, k in [("box", box), ("Dx", Dx), ("Dy", Dy)]:
        t = time.time(); four = conv_four_loops(im, k); t4 = time.time() - t
        t = time.time(); two = conv_two_loops(im, k); t2 = time.time() - t
        t = time.time(); ref = convolve2d(im, k, mode="same", boundary="fill", fillvalue=0); ts = time.time() - t
        print(f"{name:4}  four loops {t4:6.2f}s   two loops {t2:5.2f}s   scipy {ts:.4f}s   "
              f"max difference from scipy: {np.abs(four - ref).max():.1e}, {np.abs(two - ref).max():.1e}")
        save(two if name == "box" else stretch(two), f"out/1_1_{name}.png")


if __name__ == "__main__":
    main()
