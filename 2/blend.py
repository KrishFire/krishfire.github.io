"""blend.py: Parts 2.3 and 2.4, Gaussian and Laplacian stacks and multiresolution blending."""

import numpy as np
from PIL import Image
from utils import load, save, blur
from hybrid import align


# ---------------------------------------------------------------- stacks

def gaussian_stack(im, levels, sigma=2):
    """Level 0 is the image. Each level after that is blurred more.

    A pyramid halves the image at every level, which doubles how blurry it
    looks. A stack keeps the full size, so to get the same effect the blur
    doubles at every level instead.
    """
    stack = [im]
    for i in range(1, levels):
        stack.append(blur(im, sigma * 2 ** (i - 1)))
    return stack


def laplacian_stack(im, levels, sigma=2):
    """Each level is the detail that disappears between two Gaussian levels.

    The last level is the blurriest Gaussian level itself, so that adding
    every level back together gives exactly the original image.
    """
    g = gaussian_stack(im, levels, sigma)
    lap = [g[i] - g[i + 1] for i in range(levels - 1)]
    lap.append(g[-1])
    return lap


# ---------------------------------------------------------------- blending

def blend(a, b, mask, levels=6, sigma=2):
    """Blend each frequency band separately, then add the bands back up.

    Fine detail gets a sharp seam (from the lightly blurred mask) and broad
    colour gets a wide, gentle seam (from the heavily blurred mask). That is
    why the result has no visible join.
    """
    la = laplacian_stack(a, levels, sigma)
    lb = laplacian_stack(b, levels, sigma)
    gm = gaussian_stack(mask, levels, sigma)
    bands = [gm[i] * la[i] + (1 - gm[i]) * lb[i] for i in range(levels)]
    return np.clip(sum(bands), 0, 1), la, lb, gm, bands


# ---------------------------------------------------------------- masks

def half_mask(im, vertical=True, split=None):
    """1 on the left side (or top side) of the split, 0 on the other side.
    The split defaults to the middle of the image."""
    h, w = im.shape[:2]
    m = np.zeros((h, w, 3))
    if vertical:
        m[:, : (w // 2 if split is None else split)] = 1
    else:
        m[: (h // 2 if split is None else split)] = 1
    return m


def ellipse_mask(im, cx, cy, rx, ry):
    """1 inside an ellipse centred at (cx, cy), given as fractions of the width and height."""
    h, w = im.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    inside = ((x - cx * w) / (rx * w)) ** 2 + ((y - cy * h) / (ry * h)) ** 2 <= 1
    return np.dstack([inside.astype(float)] * 3)


def face_mask(im, left_eye, right_eye):
    """An oval over the face, sized and placed from where the two eyes are."""
    h, w = im.shape[:2]
    d = np.linalg.norm(np.array(right_eye) - np.array(left_eye))     # eye distance sets the size
    cx = (left_eye[0] + right_eye[0]) / 2
    cy = (left_eye[1] + right_eye[1]) / 2 + 0.3 * d                 # a face's centre sits below the eyes
    return ellipse_mask(im, cx / w, cy / h, 0.85 * d / w, 1.2 * d / h)


def mask_from_file(path, like):
    """Load a black and white mask you drew yourself (white = first image)."""
    h, w = like.shape[:2]
    m = Image.open(path).convert("L").resize((w, h))
    m = (np.asarray(m) > 127).astype(float)
    return np.dstack([m] * 3)


# ---------------------------------------------------------------- colour (bells and whistles)

def to_gray(im):
    g = im.mean(axis=2)
    return np.dstack([g, g, g])


def match_colors(src, ref, mask):
    """Shift each colour channel of src so that, inside the mask, its average and
    spread match ref. This evens out skin tone and lighting before blending, so
    the seam has less colour difference to hide."""
    inside = mask[:, :, 0] > 0.5
    out = src.copy()
    for c in range(3):
        s, r = src[:, :, c][inside], ref[:, :, c][inside]
        out[:, :, c] = (src[:, :, c] - s.mean()) / (s.std() + 1e-8) * r.std() + r.mean()
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------- pictures

def show(band):
    """Laplacian levels are small and can be negative. Stretch them so they are visible."""
    return np.clip(0.5 + band / (4 * np.abs(band).std() + 1e-8), 0, 1)


def save_stacks(name, im, levels=6):
    g = gaussian_stack(im, levels)
    lap = laplacian_stack(im, levels)
    save(np.hstack(g), f"out/2_3_{name}_gaussian_stack.png")
    save(np.hstack([show(x) for x in lap[:-1]] + [lap[-1]]), f"out/2_3_{name}_laplacian_stack.png")


def figure_342(name, la, lb, gm, bands, rows=(0, 2, 4)):
    """Recreate Szeliski figure 3.42: left column is image A masked, middle is image B
    masked, right is the two added. Bottom row is the full reconstruction."""
    grid = []
    for i in rows:
        left = gm[i] * la[i]
        mid = (1 - gm[i]) * lb[i]
        grid.append(np.hstack([show(left), show(mid), show(left + mid)]))
    total_a = sum(gm[i] * la[i] for i in range(len(la)))
    total_b = sum((1 - gm[i]) * lb[i] for i in range(len(lb)))
    grid.append(np.hstack([np.clip(total_a, 0, 1), np.clip(total_b, 0, 1),
                           np.clip(total_a + total_b, 0, 1)]))
    save(np.vstack(grid), f"out/2_4_{name}_figure342.png")


# ---------------------------------------------------------------- main

def aligned_faces(path_a, eyes_a, path_b, eyes_b):
    """Line face b up with face a, and return both crops plus a's eyes in the cropped frame."""
    a = load(path_a)
    b = load(path_b)
    a, b, (top, left) = align(a, np.array(eyes_a, float), b, np.array(eyes_b, float))
    eyes = [(x - left, y - top) for x, y in eyes_a]
    return a, b, eyes


def main():
    # ---------------- Part 2.3: stacks for the apple and the orange
    apple = load("data/spline/apple.jpeg")
    orange = load("data/spline/orange.jpeg")
    save_stacks("apple", apple)
    save_stacks("orange", orange)

    # ---------------- Part 2.4: the oraple
    m = half_mask(apple)
    oraple, la, lb, gm, bands = blend(apple, orange, m)
    save(oraple, "out/2_4_oraple.png")
    save(m * apple + (1 - m) * orange, "out/2_4_oraple_no_blend.png")
    figure_342("oraple", la, lb, gm, bands)

    ME_EYES = [(305, 258), (447, 257)]
    DEREK_EYES = [(232, 268), (342, 260)]

    # ---------------- custom 1: straight seam, half me and half Derek
    me, derek, eyes = aligned_faces("data/selfie.jpg", ME_EYES, "data/hybrid/DerekPicture.jpg", DEREK_EYES)
    split = int((eyes[0][0] + eyes[1][0]) / 2)          # cut down the middle of the nose
    m = half_mask(me, split=split)
    half, *_ = blend(me, derek, m)
    save(me, "out/2_4_seam_input_me.png")
    save(derek, "out/2_4_seam_input_derek.png")
    save(m * me + (1 - m) * derek, "out/2_4_seam_no_blend.png")
    save(half, "out/2_4_seam_blend.png")

    # ---------------- custom 2: irregular mask, my face on Derek
    m = face_mask(me, *eyes)
    save(m, "out/2_4_swap_mask.png")
    swap, la, lb, gm, bands = blend(me, derek, m)
    save(m * me + (1 - m) * derek, "out/2_4_swap_no_blend.png")
    save(swap, "out/2_4_swap_blend.png")
    figure_342("swap", la, lb, gm, bands)

    # ---------------- bells and whistles: colour
    save(blend(to_gray(me), to_gray(derek), m)[0], "out/2_4_swap_gray.png")
    me_matched = match_colors(me, derek, m)
    save(blend(me_matched, derek, m)[0], "out/2_4_swap_color_matched.png")
    save(blend(to_gray(apple), to_gray(orange), half_mask(apple))[0], "out/2_4_oraple_gray.png")


if __name__ == "__main__":
    main()
