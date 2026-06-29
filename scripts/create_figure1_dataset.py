"""
Generate Figure 1 for the IEEE Access manuscript.

The figure uses validation tiles whose original and refined SOS masks differ,
so the correction overlay is visually meaningful. SOS masks encode oil as
dark pixels and background as light pixels.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SAMPLES = [
    ("sentinel_422.png", "S1-422"),
    ("sentinel_491.png", "S1-491"),
    ("sentinel_538.png", "S1-538"),
]

PANEL = 300
OUT_PATHS = [
    Path("figures/figure1_dataset_samples.png"),
]

GREEN = np.array([35, 205, 85], dtype=np.uint8)
RED = np.array([235, 40, 40], dtype=np.uint8)
AMBER = np.array([245, 158, 11], dtype=np.uint8)
CYAN = np.array([45, 190, 230], dtype=np.uint8)
DARK = np.array([18, 22, 28], dtype=np.uint8)
BORDER = (38, 38, 38)


def _font(name, size):
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _load_sar(path):
    image = Image.open(path).convert("RGB").resize(
        (PANEL, PANEL), Image.Resampling.BILINEAR
    )
    arr = np.asarray(image).astype(np.float32)
    out = np.empty_like(arr)
    for channel in range(3):
        lo, hi = np.percentile(arr[..., channel], [1, 99])
        if hi <= lo:
            out[..., channel] = arr[..., channel]
        else:
            out[..., channel] = np.clip(
                (arr[..., channel] - lo) * 255.0 / (hi - lo), 0, 255
            )
    return out.astype(np.uint8)


def _load_oil_mask(path):
    mask = Image.open(path).convert("L").resize(
        (PANEL, PANEL), Image.Resampling.NEAREST
    )
    return np.asarray(mask) < 128


def _color_mask(mask, color):
    arr = np.zeros((PANEL, PANEL, 3), dtype=np.uint8)
    arr[:] = DARK
    arr[mask] = color
    return arr


def _overlay_corrections(sar, original, refined):
    base = sar.copy().astype(np.float32)
    unchanged = refined & original
    added = refined & ~original
    removed = original & ~refined

    alpha_unchanged = 0.34
    alpha_change = 0.82
    base[unchanged] = base[unchanged] * (1 - alpha_unchanged) + CYAN * alpha_unchanged
    base[added] = base[added] * (1 - alpha_change) + GREEN * alpha_change
    base[removed] = base[removed] * (1 - alpha_change) + RED * alpha_change
    return np.clip(base, 0, 255).astype(np.uint8)


def create_figure1_dataset_comparison():
    cols = 4
    rows = len(SAMPLES)
    header_h = 42
    label_w = 74
    gap = 18
    outer = 18
    footer_h = 34

    width = outer * 2 + label_w + cols * PANEL + (cols - 1) * gap
    height = outer * 2 + header_h + rows * PANEL + (rows - 1) * gap + footer_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    font_title = _font("arialbd.ttf", 25)
    font_row = _font("arialbd.ttf", 22)
    font_small = _font("arial.ttf", 20)

    headers = ["SAR input", "Original SOS", "Refined-SOS", "Corrections"]
    for col, title in enumerate(headers):
        x = outer + label_w + col * (PANEL + gap)
        bbox = draw.textbbox((0, 0), title, font=font_title)
        draw.text(
            (x + (PANEL - (bbox[2] - bbox[0])) / 2, outer + 4),
            title,
            fill=(20, 20, 20),
            font=font_title,
        )

    for row, (filename, row_label) in enumerate(SAMPLES):
        stem = filename.replace("sentinel_", "").replace(".png", "")
        sar = _load_sar(Path("dataset/images/images/val") / filename)
        refined = _load_oil_mask(Path("dataset/masks/masks/val") / filename)
        original = _load_oil_mask(Path("dataset_orig/test/sentinel/label") / f"{stem}.png")

        panels = [
            sar,
            _color_mask(original, AMBER),
            _color_mask(refined, CYAN),
            _overlay_corrections(sar, original, refined),
        ]

        y = outer + header_h + row * (PANEL + gap)
        bbox = draw.textbbox((0, 0), row_label, font=font_row)
        draw.text(
            (outer + (label_w - (bbox[2] - bbox[0])) / 2, y + (PANEL - (bbox[3] - bbox[1])) / 2),
            row_label,
            fill=(30, 30, 30),
            font=font_row,
        )

        for col, panel in enumerate(panels):
            x = outer + label_w + col * (PANEL + gap)
            canvas.paste(Image.fromarray(panel), (x, y))
            draw.rectangle([x, y, x + PANEL - 1, y + PANEL - 1], outline=BORDER, width=2)

    legend_y = height - outer - 23
    legend_x = outer + label_w
    legend_items = [
        ((45, 190, 230), "unchanged oil"),
        ((35, 205, 85), "added oil"),
        ((235, 40, 40), "removed label"),
    ]
    x = legend_x
    for color, text in legend_items:
        draw.rectangle([x, legend_y, x + 22, legend_y + 16], fill=color, outline=(40, 40, 40))
        draw.text((x + 30, legend_y - 3), text, fill=(25, 25, 25), font=font_small)
        x += 210

    for out_path in OUT_PATHS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, quality=95)
        print(f"Created: {out_path}")


if __name__ == "__main__":
    create_figure1_dataset_comparison()
