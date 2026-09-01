#!/usr/bin/env python3
"""
build_media.py — turn the raw camera files in _raw/ into the web assets in 0/media/,
then patch the EXIF captions and the contact sheet into 0/index.html.

Drop photos here (any mix of .heic/.jpg/.jpeg/.png, any filenames):

    _raw/part1/   exactly 2 photos  -> close-up selfie, then the stepped-back zoomed one
    _raw/part2/   exactly 2 photos  -> far+zoomed building, then close+wide building
    _raw/part3/   4-8+ photos       -> the dolly-zoom sequence, in shooting order

Ordering is taken from EXIF capture time when available, else filename.
Use --swap1 / --swap2 to flip a pair if the capture order was the other way round.

Handheld dolly-zoom frames always wobble. --stabilise locks the subject so only
the background moves; it rescales whole frames uniformly, so the effect itself is
untouched. Pass the subject box as frame fractions if the default misses it.

    python3 build_media.py --swap1 --boomerang --stabilise
"""

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "_raw"
OUT = ROOT / "0" / "media"
PAGE = ROOT / "0" / "index.html"

EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

MAX_EDGE = 1800          # long edge of the full-size web images
JPEG_Q = 84
GIF_WIDTH = 720          # source width; displayed at 600 CSS px so it stays sharp
GIF_MS = 220             # per-frame duration
THUMB_EDGE = 500


# ----------------------------------------------------------------- metadata

_EXIF_CACHE = {}


def exif_of(path: Path) -> dict:
    """Full EXIF as a tag-name dict. HEIC is routed through sips, which keeps EXIF."""
    key = str(path)
    if key in _EXIF_CACHE:
        return _EXIF_CACHE[key]

    from PIL import ExifTags

    src = path
    tmp = None
    if path.suffix.lower() in (".heic", ".heif"):
        tmp = OUT / f".exif-{path.stem}.jpg"
        subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(tmp)],
                       capture_output=True)
        src = tmp

    d = {}
    try:
        ex = Image.open(src).getexif()
        d = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
        d.update({ExifTags.TAGS.get(k, k): v for k, v in ex.get_ifd(0x8769).items()})
    except Exception:
        pass
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)

    _EXIF_CACHE[key] = d
    return d


def capture_time(path: Path):
    v = exif_of(path).get("DateTimeOriginal") or exif_of(path).get("DateTime")
    if v:
        try:
            return dt.datetime.strptime(str(v), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass
    return dt.datetime.fromtimestamp(path.stat().st_mtime)


def photo_facts(path: Path):
    """Human-readable EXIF line: focal length (35mm equiv), aperture, shutter, ISO, body."""
    d = exif_of(path)
    bits = []

    f35, fnat = d.get("FocalLengthIn35mmFilm"), d.get("FocalLength")
    if f35:
        bits.append(f"{float(f35):.0f}mm equiv.")
    elif fnat:
        bits.append(f"{float(fnat):.1f}mm")

    if d.get("FNumber"):
        bits.append(f"f/{float(d['FNumber']):.1f}")

    if d.get("ExposureTime"):
        e = float(d["ExposureTime"])
        bits.append(f"1/{round(1/e)}s" if 0 < e < 1 else f"{e:g}s")

    iso = d.get("ISOSpeedRatings") or d.get("PhotographicSensitivity")
    if iso:
        bits.append(f"ISO {int(iso)}")

    if d.get("Model"):
        bits.append(str(d["Model"]).strip())

    return " · ".join(bits)


# ----------------------------------------------------------------- images

def load(path: Path) -> Image.Image:
    """Open any input, going through sips for HEIC. Returns an upright RGB image."""
    if path.suffix.lower() in (".heic", ".heif"):
        tmp = OUT / f".tmp-{path.stem}.jpg"
        subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "100",
                        str(path), "--out", str(tmp)],
                       check=True, capture_output=True)
        im = Image.open(tmp)
        im.load()
        tmp.unlink(missing_ok=True)
    else:
        im = Image.open(path)
        im.load()

    # honour the orientation tag so portrait shots aren't sideways
    try:
        from PIL import ImageOps
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im.convert("RGB")


def fit(im: Image.Image, max_edge: int) -> Image.Image:
    w, h = im.size
    s = max_edge / max(w, h)
    if s >= 1:
        return im
    return im.resize((round(w * s), round(h * s)), Image.LANCZOS)


def save_jpeg(im: Image.Image, dest: Path, max_edge=MAX_EDGE, q=JPEG_Q):
    fit(im, max_edge).save(dest, "JPEG", quality=q, optimize=True, progressive=True)
    return dest


def collect(sub: str):
    d = RAW / sub
    files = [p for p in d.iterdir() if p.suffix.lower() in EXTS and not p.name.startswith(".")]
    files.sort(key=lambda p: (capture_time(p), p.name))
    return files


# ----------------------------------------------------------------- stabilise

def stabilise(ims, box):
    """Lock the subject so only the background moves.

    Handheld dolly-zoom stills always wobble: the subject drifts across the frame
    and its size creeps, because you can't nail "same size" by eye while walking
    backwards. This finds the subject via SIFT and applies one uniform similarity
    (scale + rotation + translation) per frame to put it back at a fixed size and
    position.

    A uniform similarity rescales the WHOLE frame, so relative sizes inside each
    frame are untouched -- the dolly-zoom effect survives exactly. It removes the
    photographer's aim, not the physics. `box` is the subject region in the first
    frame as (x0, y0, x1, y1) fractions.

    Returns aligned PIL images, all cropped to the largest window valid in every
    frame. Falls back to the originals (with a warning) if tracking fails.
    """
    import cv2

    arr = [cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR) for im in ims]
    if len({a.shape for a in arr}) != 1:
        print("!! frames differ in size; skipping stabilisation")
        return ims

    grey = [cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) for a in arr]
    h, w = grey[0].shape
    sift = cv2.SIFT_create(nfeatures=15000)

    m = np.zeros((h, w), np.uint8)
    m[int(h*box[1]):int(h*box[3]), int(w*box[0]):int(w*box[2])] = 255
    k1, d1 = sift.detectAndCompute(grey[0], m)
    if d1 is None or len(k1) < 12:
        print(f"!! only {len(k1)} keypoints in the subject box; skipping stabilisation")
        return ims

    inv = [np.float32([[1, 0, 0], [0, 1, 0]])]
    for i in range(1, len(arr)):
        k2, d2 = sift.detectAndCompute(grey[i], None)
        good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2)
                if a.distance < 0.72 * b.distance]
        if len(good) < 8:
            print(f"!! frame {i+1}: only {len(good)} subject matches; skipping stabilisation")
            return ims
        p1 = np.float32([k1[a.queryIdx].pt for a in good]).reshape(-1, 1, 2)
        p2 = np.float32([k2[a.trainIdx].pt for a in good]).reshape(-1, 1, 2)
        T, _ = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC,
                                           ransacReprojThreshold=3.0, maxIters=25000)
        if T is None:
            print(f"!! frame {i+1}: no transform; skipping stabilisation")
            return ims
        inv.append(np.linalg.inv(np.vstack([T, [0, 0, 1]]))[:2])

    warped = [cv2.warpAffine(arr[i], inv[i], (w, h), flags=cv2.INTER_LANCZOS4)
              for i in range(len(arr))]

    # Largest window that is real image in EVERY frame. Take the biggest
    # axis-aligned rectangle *inside* each warped quad -- its bounding box would
    # be too generous and would leave black wedges along the edges.
    x0, y0, x1, y1 = 0.0, 0.0, float(w), float(h)
    quad = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    for t in inv:
        tl, tr, br, bl = cv2.transform(quad, t).reshape(-1, 2)
        x0, x1 = max(x0, tl[0], bl[0]), min(x1, tr[0], br[0])
        y0, y1 = max(y0, tl[1], tr[1]), min(y1, bl[1], br[1])
    x0, y0, x1, y1 = int(x0 + 6), int(y0 + 6), int(x1 - 6), int(y1 - 6)
    print(f"   stabilised: subject locked, kept {(x1-x0)*(y1-y0)/(w*h)*100:.0f}% of frame")

    return [Image.fromarray(cv2.cvtColor(v[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
            for v in warped]


# ----------------------------------------------------------------- html patch

def patch_page(exif: dict, frames: list):
    html = PAGE.read_text()

    for slot, text in exif.items():
        if not text:
            continue
        html = re.sub(
            rf'(<span class="exif" data-exif="{re.escape(slot)}">)(.*?)(</span>)',
            lambda m: m.group(1) + text + m.group(3),
            html, flags=re.S)

    sheet = "\n".join(
        f'        <a href="media/{f}" target="_blank" rel="noopener">'
        f'<img src="media/{t}" alt="Dolly zoom frame {i}" loading="lazy">'
        f'<span class="n">{i:02d}</span></a>'
        for i, (f, t) in enumerate(frames, 1))

    html = re.sub(r"(<!-- CONTACT_SHEET_START -->)(.*?)(\s*<!-- CONTACT_SHEET_END -->)",
                  lambda m: m.group(1) + "\n" + sheet + "\n" + m.group(3).lstrip("\n"),
                  html, flags=re.S)

    PAGE.write_text(html)


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap1", action="store_true", help="flip the Part 1 pair")
    ap.add_argument("--swap2", action="store_true", help="flip the Part 2 pair")
    ap.add_argument("--reverse3", action="store_true", help="reverse the dolly-zoom order")
    ap.add_argument("--boomerang", action="store_true", help="play the GIF forwards then back")
    ap.add_argument("--ms", type=int, default=GIF_MS, help="GIF frame duration in ms")
    ap.add_argument("--stabilise", "--stabilize", dest="stabilise", metavar="x0,y0,x1,y1",
                    nargs="?", const="0.43,0.455,0.585,0.635",
                    help="lock the dolly-zoom subject; optional box as frame fractions")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    exif = {}

    # ---- Part 1 & Part 2 pairs -------------------------------------------
    for part, slots, swap in (("part1", ("close", "far"), args.swap1),
                              ("part2", ("far", "close"), args.swap2)):
        files = collect(part)
        if len(files) < 2:
            print(f"!! _raw/{part}/ has {len(files)} photo(s); need 2. Skipping.")
            continue
        if len(files) > 2:
            print(f"!! _raw/{part}/ has {len(files)} photos; using the first 2 by capture time.")
            files = files[:2]
        if swap:
            files.reverse()
        for src, slot in zip(files, slots):
            name = f"{part}-{slot}.jpg"
            save_jpeg(load(src), OUT / name)
            exif[f"{part}-{slot}"] = photo_facts(src)
            print(f"   {src.name}  ->  0/media/{name}   [{exif[f'{part}-{slot}']}]")

    # ---- Part 3 sequence + GIF -------------------------------------------
    frames = []
    files = collect("part3")
    if args.reverse3:
        files.reverse()

    if len(files) < 2:
        print(f"!! _raw/part3/ has {len(files)} photo(s); need at least 4. Skipping GIF.")
    else:
        if len(files) < 4:
            print(f"!! only {len(files)} dolly-zoom frames; the spec asks for 4-8+.")
        stills = []
        for i, src in enumerate(files, 1):
            im = fit(load(src), MAX_EDGE)
            full, thumb = f"part3-{i:02d}.jpg", f"part3-{i:02d}-thumb.jpg"
            save_jpeg(im, OUT / full)
            save_jpeg(im, OUT / thumb, max_edge=THUMB_EDGE, q=80)
            frames.append((full, thumb))
            stills.append(im)
            print(f"   {src.name}  ->  0/media/{full}")

        # the stills stay exactly as shot; only the GIF gets aligned
        gif_src = stills
        if args.stabilise:
            box = tuple(float(v) for v in args.stabilise.split(","))
            gif_src = stabilise(stills, box)

        gif_frames = [fit(im, round(GIF_WIDTH * max(im.size) / im.size[0])).convert(
            "P", palette=Image.ADAPTIVE, colors=128) for im in gif_src]

        base = gif_frames[0].size
        gif_frames = [f if f.size == base else f.resize(base, Image.LANCZOS) for f in gif_frames]

        seq = gif_frames + gif_frames[-2:0:-1] if args.boomerang else gif_frames
        gif = OUT / "dolly-zoom.gif"
        seq[0].save(gif, save_all=True, append_images=seq[1:],
                    duration=args.ms, loop=0, optimize=True, disposal=2)
        print(f"   -> 0/media/dolly-zoom.gif  ({len(seq)} frames, "
              f"{gif.stat().st_size/1_048_576:.1f} MB)")

    patch_page(exif, frames)
    print("\nPatched 0/index.html (EXIF captions + contact sheet).")


if __name__ == "__main__":
    sys.exit(main())
