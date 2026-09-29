"""utils.py: small helpers shared by every part of the project."""

import numpy as np
import cv2
from PIL import Image, ImageOps
from scipy.signal import convolve2d


def load(path, gray=False, max_side=800):
    """Read an image as floats between 0 and 1, shrunk so the long side is at most max_side."""
    im = ImageOps.exif_transpose(Image.open(path))   # phone photos store their rotation as a tag
    im = im.convert("L" if gray else "RGB")
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    return np.asarray(im).astype(np.float64) / 255.0


def save(im, path):
    """Save a float image. Values outside 0 to 1 are clipped first."""
    im = np.clip(im, 0, 1)
    Image.fromarray((im * 255).round().astype(np.uint8)).save(path)


def stretch(im):
    """Rescale so the smallest value becomes 0 and the largest becomes 1.
    Used to display things like derivatives, which can be negative."""
    return (im - im.min()) / (im.max() - im.min())


def gaussian_1d(sigma):
    """A 1D Gaussian as a column vector. The width is about 6 sigma so the tails are included."""
    ksize = int(6 * sigma) | 1          # "| 1" forces an odd number so there is a centre pixel
    return cv2.getGaussianKernel(ksize, sigma)


def gaussian_2d(sigma):
    """Outer product of the 1D Gaussian with itself gives the 2D Gaussian."""
    g = gaussian_1d(sigma)
    return g @ g.T


def blur(im, sigma):
    """Gaussian blur. Works on gray or colour images.

    A 2D Gaussian can be split into a vertical 1D blur followed by a horizontal
    1D blur, which gives the same answer much faster. With a 31x31 kernel that is
    62 multiplications per pixel instead of 961.
    """
    g = gaussian_1d(sigma)
    if im.ndim == 3:
        return np.dstack([blur(im[:, :, c], sigma) for c in range(im.shape[2])])
    out = convolve2d(im, g, mode="same", boundary="symm")
    return convolve2d(out, g.T, mode="same", boundary="symm")
