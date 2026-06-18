from __future__ import annotations

import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from tkinter import (
    BOTH,
    BooleanVar,
    Checkbutton,
    DISABLED,
    END,
    LEFT,
    NORMAL,
    RIGHT,
    X,
    Button,
    DoubleVar,
    Frame,
    Label,
    Listbox,
    Scale,
    StringVar,
    Tk,
    filedialog,
    messagebox,
)

from PIL import Image, ImageDraw, ImageFont, ImageTk


DEFAULT_MODEL = Path("runs/detect/runs/rice_detection/weights/best.pt")
IMAGE_TYPES = [
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
    ("All files", "*.*"),
]
MODEL_TYPES = [("YOLO model", "*.pt"), ("All files", "*.*")]
BOX_COLORS = [
    "#e53935",
    "#1e88e5",
    "#43a047",
    "#fb8c00",
    "#8e24aa",
    "#00acc1",
    "#6d4c41",
]


def relaunch_with_python311_if_needed() -> None:
    if sys.version_info[:2] == (3, 11):
        return

    script = Path(__file__).resolve()
    command_options = [
        ["py", "-3.11", str(script), *sys.argv[1:]],
        ["python", str(script), *sys.argv[1:]],
    ]

    for command in command_options:
        try:
            subprocess.Popen(command, cwd=str(script.parent))
            sys.exit(0)
        except OSError:
            continue


class RicePredictionGUI:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Prediksi Jenis Beras")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)

        self.model_path = StringVar(value=str(DEFAULT_MODEL))
        self.image_path = StringVar(value="")
        self.status = StringVar(value="Pilih gambar untuk memulai prediksi.")
        self.confidence = DoubleVar(value=0.25)
        self.tight_box = BooleanVar(value=True)

        self.model: Any | None = None
        self.loaded_model_path: str | None = None
        self.original_image: Image.Image | None = None
        self.result_image: Image.Image | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.best_class_name: str | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        top = Frame(self.root, padx=12, pady=10)
        top.pack(fill=X)

        Button(top, text="Pilih Model", command=self.choose_model, width=14).pack(side=LEFT)
        Label(top, textvariable=self.model_path, anchor="w").pack(side=LEFT, fill=X, expand=True, padx=8)

        image_row = Frame(self.root, padx=12, pady=4)
        image_row.pack(fill=X)

        Button(image_row, text="Pilih Gambar", command=self.choose_image, width=14).pack(side=LEFT)
        Label(image_row, textvariable=self.image_path, anchor="w").pack(side=LEFT, fill=X, expand=True, padx=8)

        control = Frame(self.root, padx=12, pady=8)
        control.pack(fill=X)

        Label(control, text="Confidence").pack(side=LEFT)
        Scale(
            control,
            from_=0.05,
            to=0.95,
            resolution=0.05,
            orient="horizontal",
            variable=self.confidence,
            length=240,
        ).pack(side=LEFT, padx=8)
        Checkbutton(control, text="Box menyesuaikan beras", variable=self.tight_box).pack(side=LEFT, padx=8)

        self.predict_button = Button(control, text="Prediksi", command=self.predict, width=14)
        self.predict_button.pack(side=LEFT, padx=4)

        self.save_button = Button(control, text="Simpan Hasil", command=self.save_result, width=14, state=DISABLED)
        self.save_button.pack(side=LEFT, padx=4)

        Button(control, text="Reset", command=self.reset, width=10).pack(side=LEFT, padx=4)

        main = Frame(self.root, padx=12, pady=8)
        main.pack(fill=BOTH, expand=True)

        preview_frame = Frame(main, bd=1, relief="solid")
        preview_frame.pack(side=LEFT, fill=BOTH, expand=True)

        self.preview_label = Label(preview_frame, text="Preview gambar", bg="#f5f5f5")
        self.preview_label.pack(fill=BOTH, expand=True)
        self.preview_label.bind("<Configure>", lambda _event: self.refresh_preview())

        side = Frame(main, width=320)
        side.pack(side=RIGHT, fill="y", padx=(12, 0))
        side.pack_propagate(False)

        Label(side, text="Hasil Deteksi", anchor="w").pack(fill=X)
        self.result_list = Listbox(side, height=18)
        self.result_list.pack(fill=BOTH, expand=True, pady=(6, 8))

        Label(side, textvariable=self.status, anchor="w", justify=LEFT, wraplength=300).pack(fill=X)

    def choose_model(self) -> None:
        path = filedialog.askopenfilename(title="Pilih model YOLO", filetypes=MODEL_TYPES)
        if path:
            self.model_path.set(path)
            self.model = None
            self.loaded_model_path = None
            self.status.set("Model dipilih. Jalankan prediksi untuk memuat model.")

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(title="Pilih gambar beras", filetypes=IMAGE_TYPES)
        if not path:
            return

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Gagal membuka gambar", str(exc))
            return

        self.image_path.set(path)
        self.original_image = image
        self.result_image = None
        self.best_class_name = None
        self.save_button.config(state=DISABLED)
        self.result_list.delete(0, END)
        self.status.set("Gambar siap diprediksi.")
        self.refresh_preview()

    def predict(self) -> None:
        image_path = self.image_path.get()
        model_path = self.model_path.get()

        if not image_path:
            messagebox.showwarning("Gambar belum dipilih", "Pilih gambar terlebih dahulu.")
            return
        if not model_path or not Path(model_path).exists():
            messagebox.showwarning("Model tidak ditemukan", "Pilih file model .pt yang valid.")
            return

        self.predict_button.config(state=DISABLED)
        self.save_button.config(state=DISABLED)
        self.status.set("Memproses prediksi...")
        self.result_list.delete(0, END)

        thread = threading.Thread(
            target=self._predict_worker,
            args=(Path(model_path), Path(image_path), float(self.confidence.get()), bool(self.tight_box.get())),
            daemon=True,
        )
        thread.start()

    def _predict_worker(self, model_path: Path, image_path: Path, confidence: float, tight_box: bool) -> None:
        try:
            from ultralytics import YOLO

            if self.model is None or self.loaded_model_path != str(model_path):
                self.model = YOLO(str(model_path))
                self.loaded_model_path = str(model_path)

            results = self.model.predict(
                source=str(image_path),
                conf=confidence,
                imgsz=416,
                max_det=10,
                verbose=False,
            )
            result = results[0]
            source_image = Image.open(image_path).convert("RGB")

            detections: list[tuple[str, float, tuple[int, int, int, int]]] = []
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                score = float(box.conf[0])
                xyxy = tuple(int(round(value)) for value in box.xyxy[0].tolist())
                if tight_box:
                    xyxy = tighten_box(source_image, xyxy)
                detections.append((str(names[class_id]), score, xyxy))

            result_image = draw_detections(source_image, detections)

            self.root.after(0, self._show_result, result_image, detections)
        except Exception as exc:
            self.root.after(0, self._show_error, self._friendly_error(exc))

    def _friendly_error(self, exc: Exception) -> str:
        message = str(exc)
        lowered = message.lower()
        if "torch" in lowered or "ultralytics" in lowered:
            return (
                "Gagal memuat Ultralytics/Torch.\n\n"
                "Command `py` di komputer ini memakai Python 3.13, sedangkan training memakai Python 3.11.\n\n"
                "Jalankan GUI dengan:\n"
                "python gui_predict.py\n"
                "atau\n"
                "py -3.11 gui_predict.py\n\n"
                f"Detail error: {message}"
            )
        return message

    def _show_result(
        self,
        image: Image.Image,
        detections: list[tuple[str, float, tuple[int, int, int, int]]],
    ) -> None:
        self.result_image = image
        self.refresh_preview()

        self.result_list.delete(0, END)
        if detections:
            for index, (class_name, score, _xyxy) in enumerate(detections, start=1):
                self.result_list.insert(END, f"{index}. {class_name} - {score:.2%}")
            best_class, best_score, _xyxy = max(detections, key=lambda item: item[1])
            self.best_class_name = best_class
            self.status.set(f"Prediksi selesai. Teratas: {best_class} ({best_score:.2%}).")
            self.save_button.config(state=NORMAL)
        else:
            self.best_class_name = None
            self.status.set("Tidak ada objek terdeteksi. Coba turunkan confidence.")

        self.predict_button.config(state=NORMAL)

    def _show_error(self, message: str) -> None:
        self.predict_button.config(state=NORMAL)
        self.status.set("Prediksi gagal.")
        messagebox.showerror("Prediksi gagal", message)

    def save_result(self) -> None:
        if self.result_image is None:
            messagebox.showwarning("Belum ada hasil", "Jalankan prediksi terlebih dahulu.")
            return

        source = Path(self.image_path.get())
        class_folder = sanitize_folder_name(self.best_class_name or "Tidak_Terdeteksi")
        output_dir = Path("hasil_prediksi") / class_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{source.stem}_prediksi_{timestamp}.jpg"

        self.result_image.save(output_path, quality=95)
        self.status.set(f"Hasil disimpan otomatis: {output_path}")
        messagebox.showinfo("Hasil disimpan", f"Gambar tersimpan di:\n{output_path.resolve()}")

    def reset(self) -> None:
        self.image_path.set("")
        self.original_image = None
        self.result_image = None
        self.best_class_name = None
        self.preview_photo = None
        self.preview_label.config(image="", text="Preview gambar")
        self.result_list.delete(0, END)
        self.save_button.config(state=DISABLED)
        self.status.set("Pilih gambar untuk memulai prediksi.")

    def refresh_preview(self) -> None:
        image = self.result_image or self.original_image
        if image is None:
            return

        width = max(1, self.preview_label.winfo_width() - 24)
        height = max(1, self.preview_label.winfo_height() - 24)
        preview = image.copy()
        preview.thumbnail((width, height), Image.Resampling.LANCZOS)

        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_label.config(image=self.preview_photo, text="")


def tighten_box(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    tight_box, _contours = segment_rice_area(image, box)
    return tight_box


def segment_rice_area(
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], list[list[tuple[int, int]]]]:
    import cv2
    import numpy as np

    width, height = image.size
    x1, y1, x2, y2 = clamp_box(box, width, height)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return (x1, y1, x2, y2), []

    crop = image.crop((x1, y1, x2, y2))
    original_crop_w, original_crop_h = crop.size
    max_process_size = 416
    scale = min(1.0, max_process_size / max(original_crop_w, original_crop_h))
    if scale < 1.0:
        resized_size = (max(1, int(original_crop_w * scale)), max(1, int(original_crop_h * scale)))
        crop = crop.resize(resized_size, Image.Resampling.BILINEAR)

    crop_rgb = np.array(crop)
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    crop_h, crop_w = crop_bgr.shape[:2]

    inset_x = max(1, int(crop_w * 0.03))
    inset_y = max(1, int(crop_h * 0.03))
    rect = (inset_x, inset_y, max(1, crop_w - 2 * inset_x), max(1, crop_h - 2 * inset_y))

    foreground = fast_rice_mask(crop_rgb)

    foreground = clean_mask(foreground)
    contour_box = mask_to_box(foreground)
    if contour_box is None:
        foreground = threshold_foreground(crop_rgb)
        foreground = clean_mask(foreground)
        contour_box = mask_to_box(foreground)

    if contour_box is None:
        return (x1, y1, x2, y2), []

    bx1, by1, bx2, by2 = contour_box
    pad = max(4, int(max(crop_w, crop_h) * 0.02))
    scale_x = original_crop_w / crop_w
    scale_y = original_crop_h / crop_h
    tight_box = clamp_box(
        (
            x1 + int((bx1 - pad) * scale_x),
            y1 + int((by1 - pad) * scale_y),
            x1 + int((bx2 + pad) * scale_x),
            y1 + int((by2 + pad) * scale_y),
        ),
        width,
        height,
    )
    contours = mask_to_global_contours(foreground, x1, y1, scale_x, scale_y)
    return tight_box, contours


def fast_rice_mask(crop_rgb: Any) -> Any:
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    min_rgb = crop_rgb.min(axis=2)
    max_rgb = crop_rgb.max(axis=2)
    channel_range = max_rgb - min_rgb

    # Rice is usually bright and low-saturation; the dataset background is mostly green.
    loose_mask = (
        ((saturation < 85) & (value > 120))
        | ((min_rgb > 125) & (channel_range < 80))
    ).astype("uint8") * 255

    area_ratio = cv2.countNonZero(loose_mask) / float(loose_mask.shape[0] * loose_mask.shape[1])
    if 0.0002 <= area_ratio <= 0.35:
        return loose_mask

    strict_mask = (
        ((saturation < 55) & (value > 150))
        | ((min_rgb > 155) & (channel_range < 55))
    ).astype("uint8") * 255
    strict_ratio = cv2.countNonZero(strict_mask) / float(strict_mask.shape[0] * strict_mask.shape[1])
    if 0.00005 <= strict_ratio <= 0.35:
        return strict_mask

    return threshold_foreground(crop_rgb)


def threshold_foreground(crop_rgb: Any) -> Any:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _threshold, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:
        mask = cv2.bitwise_not(mask)
    return mask


def clean_mask(mask: Any) -> Any:
    import cv2
    import numpy as np

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def mask_to_box(mask: Any) -> tuple[int, int, int, int] | None:
    import cv2

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    crop_area = mask.shape[0] * mask.shape[1]
    contours = [contour for contour in contours if cv2.contourArea(contour) > crop_area * 0.0008]
    if not contours:
        return None

    areas = [cv2.contourArea(contour) for contour in contours]
    largest_area = max(areas)
    min_area = max(crop_area * 0.001, largest_area * 0.18)
    contours = [
        contour
        for contour, area in zip(contours, areas)
        if area >= min_area
    ]
    if not contours:
        return None

    x_values: list[int] = []
    y_values: list[int] = []
    x2_values: list[int] = []
    y2_values: list[int] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        x_values.append(x)
        y_values.append(y)
        x2_values.append(x + width)
        y2_values.append(y + height)

    return min(x_values), min(y_values), max(x2_values), max(y2_values)


def mask_to_global_contours(mask: Any, offset_x: int, offset_y: int, scale_x: float, scale_y: float) -> list[list[tuple[int, int]]]:
    import cv2

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = mask.shape[0] * mask.shape[1]
    output: list[list[tuple[int, int]]] = []

    for contour in contours:
        if cv2.contourArea(contour) <= crop_area * 0.002:
            continue

        epsilon = 0.003 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = [
            (offset_x + int(point[0][0] * scale_x), offset_y + int(point[0][1] * scale_y))
            for point in approx
        ]
        if len(points) >= 3:
            output.append(points)

    return output


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
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

        for offset in range(line_width):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)

        label_box = draw.textbbox((x1, y1), label, font=font)
        padding_x = max(10, font_size // 3)
        padding_y = max(6, font_size // 5)
        label_width = label_box[2] - label_box[0] + padding_x * 2
        label_height = label_box[3] - label_box[1] + padding_y * 2
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
        text_x = label_x + padding_x
        text_y = label_y + padding_y
        draw.text((text_x + 2, text_y + 2), label, fill="black", font=font)
        draw.text((text_x, text_y), label, fill="white", font=font)

    return output


def load_label_font(size: int) -> ImageFont.ImageFont:
    font_candidates = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def sanitize_folder_name(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    cleaned = "".join("_" if char in invalid_chars else char for char in name).strip()
    return cleaned or "Tidak_Terdeteksi"


def main() -> None:
    relaunch_with_python311_if_needed()
    root = Tk()
    RicePredictionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
