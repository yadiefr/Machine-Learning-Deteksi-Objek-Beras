from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


BOX_COLORS = [
    "#e53935",
    "#1e88e5",
    "#43a047",
    "#fb8c00",
    "#8e24aa",
    "#00acc1",
    "#6d4c41",
]


def tighten_box(
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    import cv2
    import numpy as np

    width, height = image.size
    x1, y1, x2, y2 = clamp_box(box, width, height)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return x1, y1, x2, y2

    crop = image.crop((x1, y1, x2, y2))
    original_width, original_height = crop.size
    scale = min(1.0, 416 / max(original_width, original_height))
    if scale < 1.0:
        crop = crop.resize(
            (max(1, int(original_width * scale)), max(1, int(original_height * scale))),
            Image.Resampling.BILINEAR,
        )

    crop_rgb = np.asarray(crop)
    mask = clean_mask(fast_rice_mask(crop_rgb))
    detected_box = mask_to_box(mask)
    if detected_box is None:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _threshold, mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
        if border.mean() > 127:
            mask = cv2.bitwise_not(mask)
        detected_box = mask_to_box(clean_mask(mask))

    if detected_box is None:
        return x1, y1, x2, y2

    bx1, by1, bx2, by2 = detected_box
    crop_height, crop_width = mask.shape[:2]
    padding = max(4, int(max(crop_width, crop_height) * 0.02))
    scale_x = original_width / crop_width
    scale_y = original_height / crop_height
    return clamp_box(
        (
            x1 + int((bx1 - padding) * scale_x),
            y1 + int((by1 - padding) * scale_y),
            x1 + int((bx2 + padding) * scale_x),
            y1 + int((by2 + padding) * scale_y),
        ),
        width,
        height,
    )


def fast_rice_mask(crop_rgb: Any) -> Any:
    import cv2

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    min_rgb = crop_rgb.min(axis=2)
    channel_range = crop_rgb.max(axis=2) - min_rgb

    loose = (
        ((saturation < 85) & (value > 120))
        | ((min_rgb > 125) & (channel_range < 80))
    ).astype("uint8") * 255
    ratio = cv2.countNonZero(loose) / float(loose.shape[0] * loose.shape[1])
    if 0.0002 <= ratio <= 0.35:
        return loose

    return (
        ((saturation < 55) & (value > 150))
        | ((min_rgb > 155) & (channel_range < 55))
    ).astype("uint8") * 255


def clean_mask(mask: Any) -> Any:
    import cv2
    import numpy as np

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def mask_to_box(mask: Any) -> tuple[int, int, int, int] | None:
    import cv2

    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    crop_area = mask.shape[0] * mask.shape[1]
    contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) > crop_area * 0.0008
    ]
    if not contours:
        return None

    largest_area = max(cv2.contourArea(contour) for contour in contours)
    minimum_area = max(crop_area * 0.001, largest_area * 0.18)
    contours = [
        contour for contour in contours if cv2.contourArea(contour) >= minimum_area
    ]
    boxes = [cv2.boundingRect(contour) for contour in contours]
    return (
        min(x for x, _y, _w, _h in boxes),
        min(y for _x, y, _w, _h in boxes),
        max(x + w for x, _y, w, _h in boxes),
        max(y + h for _x, y, _w, h in boxes),
    )


def clamp_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def draw_detections(
    image: Image.Image,
    detections: list[tuple[str, float, tuple[int, int, int, int]]],
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font_size = max(28, int(min(image.size) * 0.025))
    line_width = max(4, int(min(image.size) * 0.005))
    font = load_label_font(font_size)

    for index, (class_name, score, box) in enumerate(detections):
        color = BOX_COLORS[index % len(BOX_COLORS)]
        x1, y1, x2, y2 = box
        label = f"{class_name} {score:.0%}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        text_box = draw.textbbox((0, 0), label, font=font)
        padding_x = max(10, font_size // 3)
        padding_y = max(6, font_size // 5)
        label_width = text_box[2] - text_box[0] + padding_x * 2
        label_height = text_box[3] - text_box[1] + padding_y * 2
        label_x = min(max(0, x1), max(0, image.width - label_width))
        label_y = y1 - label_height - line_width
        if label_y < 0:
            label_y = min(image.height - label_height, y1 + line_width)

        draw.rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            fill=color,
            outline="black",
            width=max(2, line_width // 2),
        )
        draw.text(
            (label_x + padding_x, label_y + padding_y),
            label,
            fill="white",
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )

    return output


def load_label_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()
