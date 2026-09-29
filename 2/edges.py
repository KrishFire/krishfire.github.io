"""edges.py: Parts 1.2 and 1.3, plus the gradient orientation bells and whistles."""

import numpy as np
from scipy.signal import convolve2d
from utils import load, save, stretch, gaussian_2d

Dx = np.array([[1, -1]])
Dy = np.array([[1], [-1]])


def conv(im, k):
    return convolve2d(im, k, mode="same", boundary="symm")


def gradients(im):
    """Return the x derivative, the y derivative and the gradient magnitude."""
    gx = conv(im, Dx)
    gy = conv(im, Dy)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return gx, gy, mag


def orientation_colors(gx, gy, mag):
    """Colour each pixel by the direction of its edge, without computing any angles.

    If the edge direction is an angle t, then cos(t) = gx / mag and sin(t) = gy / mag.
    A colour wheel can be built straight from those two numbers: red, green and blue
    are three cosine waves 120 degrees apart. Expanding cos(t - 120) with the angle
    subtraction rule only needs cos(t) and sin(t), so no arctan is needed.
    Brightness is the gradient magnitude, so flat regions stay dark.
    """
    safe = np.where(mag == 0, 1, mag)
    c = gx / safe
    s = gy / safe
    root3 = np.sqrt(3) / 2
    r = 0.5 + 0.5 * c
    g = 0.5 + 0.5 * (-0.5 * c + root3 * s)
    b = 0.5 + 0.5 * (-0.5 * c - root3 * s)
    brightness = np.clip(mag / np.percentile(mag, 99), 0, 1)
    return np.dstack([r, g, b]) * brightness[:, :, None]


def main():
    im = load("data/cameraman.png", gray=True)

    # ---------------- Part 1.2: plain finite differences
    gx, gy, mag = gradients(im)
    save(stretch(gx), "out/1_2_dx.png")
    save(stretch(gy), "out/1_2_dy.png")
    save(stretch(mag), "out/1_2_mag.png")
    for t in [0.10, 0.15, 0.20, 0.25, 0.30]:          # try a few thresholds and look at them
        save((mag > t).astype(float), f"out/1_2_edges_{t:.2f}.png")

    # ---------------- Part 1.3a: blur first, then take derivatives (two convolutions)
    G = gaussian_2d(sigma=1.5)
    smooth = conv(im, G)
    sgx, sgy, smag = gradients(smooth)
    save(smooth, "out/1_3_blurred.png")
    save(stretch(sgx), "out/1_3_dx.png")
    save(stretch(sgy), "out/1_3_dy.png")
    save(stretch(smag), "out/1_3_mag.png")
    for t in [0.04, 0.06, 0.08, 0.10]:
        save((smag > t).astype(float), f"out/1_3_edges_{t:.2f}.png")

    # ---------------- Part 1.3b: build the DoG filters, then use one convolution each
    DoGx = convolve2d(G, Dx)            # "full" mode keeps the whole filter
    DoGy = convolve2d(G, Dy)
    save(stretch(DoGx), "out/1_3_DoGx_filter.png")
    save(stretch(DoGy), "out/1_3_DoGy_filter.png")
    dgx = conv(im, DoGx)
    dgy = conv(im, DoGy)
    dmag = np.sqrt(dgx ** 2 + dgy ** 2)
    save(stretch(dmag), "out/1_3_DoG_mag.png")

    # the two routes should agree, apart from a tiny difference at the image border
    h, w = im.shape
    inner = (slice(10, h - 10), slice(10, w - 10))
    print("biggest difference between blur-then-derive and DoG:",
          f"{np.abs(smag[inner] - dmag[inner]).max():.2e}")

    # ---------------- Bells and whistles: gradient orientation
    save(orientation_colors(sgx, sgy, smag), "out/1_bw_orientation.png")



if __name__ == "__main__":
    main()
