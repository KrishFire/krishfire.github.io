# krishfire.github.io

Krish Tandon's portfolio for **CS180/280A — Intro to Computer Vision and Computational
Photography** (UC Berkeley, Fall 2026).

Live at **https://krishfire.github.io**

## Layout

```
/                     portfolio landing page
├─ index.html
├─ style.css          shared stylesheet, imported as /style.css
├─ 0/                 Project 0 — Becoming Friends with Your Camera
│  ├─ index.html
│  └─ media/          web-ready jpgs + dolly-zoom.gif  (committed)
├─ _raw/              original camera files            (git-ignored)
│  ├─ part1/  part2/  part3/
└─ build_media.py     _raw/ -> 0/media/ + patches EXIF captions into 0/index.html
```

## Adding photos

Drop the originals (HEIC or JPEG, any filenames) into `_raw/`:

- `_raw/part1/` — 2 photos: the close-up selfie and the stepped-back zoomed one
- `_raw/part2/` — 2 photos: the far/zoomed building and the close/wide building
- `_raw/part3/` — the dolly-zoom sequence, 4–8+ frames

Then:

```bash
python3 build_media.py          # --swap1 / --swap2 to flip a pair, --boomerang for the gif
python3 -m http.server 8080     # preview at http://localhost:8080
```

`build_media.py` converts HEIC via `sips`, fixes orientation, resizes, writes the GIF,
and injects the real EXIF (focal length, aperture, shutter, ISO) into the captions.
