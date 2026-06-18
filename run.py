from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = [
    "Beras_Gundukan",
    "Beras_Patahan",
    "Beras_Utuh",
    "IR42",
    "IR64",
    "Ketan",
    "Pandan",
]


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_dataset(dataset_dir: Path) -> dict[str, list[Path]]:
    classes = [path.name for path in sorted(dataset_dir.iterdir()) if path.is_dir()]
    if not classes:
        raise FileNotFoundError(f"Tidak ada subfolder kelas di {dataset_dir}")

    by_class: dict[str, list[Path]] = {}
    for class_name in classes:
        files = image_files(dataset_dir / class_name)
        if files:
            by_class[class_name] = files

    if not by_class:
        raise FileNotFoundError(f"Tidak ada gambar di {dataset_dir}")
    return by_class


def split_items(
    items: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = max(1, int(total * train_ratio))
    val_count = max(1, int(total * val_ratio)) if total >= 3 else 0

    if train_count + val_count >= total and total > 1:
        val_count = max(0, total - train_count - 1)

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_yolo_label(path: Path, class_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Dataset asli hanya punya label folder kelas, jadi bounding box memakai seluruh frame.
    path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")


def prepare_dataset(
    source: Path,
    output: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Path:
    by_class = collect_dataset(source)
    class_names = sorted(by_class)
    class_to_id = {name: index for index, name in enumerate(class_names)}

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    rows: list[dict[str, str]] = []
    split_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for class_name, files in by_class.items():
        splits = split_items(files, train_ratio, val_ratio, seed)
        for split_name, split_files in splits.items():
            for src in split_files:
                dst_name = f"{class_name}__{src.name}"
                image_dst = output / "images" / split_name / dst_name
                label_dst = output / "labels" / split_name / f"{Path(dst_name).stem}.txt"

                safe_copy(src, image_dst)
                write_yolo_label(label_dst, class_to_id[class_name])

                width, height = Image.open(src).size
                rows.append(
                    {
                        "split": split_name,
                        "class": class_name,
                        "image": str(image_dst),
                        "label": str(label_dst),
                        "width": str(width),
                        "height": str(height),
                    }
                )
                split_counter[split_name][class_name] += 1

    data_yaml = {
        "path": str(output.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    data_yaml_path = output / "data.yaml"
    data_yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    csv_path = output / "dataset_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "class", "image", "label", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset YOLO selesai dibuat: {output}")
    print(f"Konfigurasi YOLO: {data_yaml_path}")
    for split_name in ["train", "val", "test"]:
        total = sum(split_counter[split_name].values())
        detail = ", ".join(
            f"{name}={split_counter[split_name][name]}" for name in class_names
        )
        print(f"{split_name}: {total} gambar ({detail})")

    return data_yaml_path


def train_model(
    data_yaml: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project: Path,
    run_name: str,
) -> Path:
    from ultralytics import YOLO

    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name=run_name,
        patience=10,
        seed=42,
        exist_ok=True,
    )
    save_dir = Path(results.save_dir)
    best_model = save_dir / "weights" / "best.pt"
    print(f"Training selesai. Model terbaik: {best_model}")
    return best_model


def evaluate_model(model_path: Path, data_yaml: Path, imgsz: int, split: str) -> None:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, split=split)
    box = metrics.box
    print(f"Evaluasi split {split}")
    print(f"mAP50-95: {box.map:.4f}")
    print(f"mAP50: {box.map50:.4f}")
    print(f"Precision: {box.mp:.4f}")
    print(f"Recall: {box.mr:.4f}")


def predict(model_path: Path, source: Path, imgsz: int, conf: float, output: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        project=str(output),
        name="predictions",
        save=True,
        exist_ok=True,
    )
    print(f"Hasil prediksi tersimpan di {output / 'predictions'}")


def write_report(output_path: Path, yolo_dir: Path, run_dir: Path | None = None) -> None:
    summary_path = yolo_dir / "dataset_summary.csv"
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                counts[row["split"]][row["class"]] += 1

    class_names = sorted({name for counter in counts.values() for name in counter})
    lines = [
        "# Laporan Akhir Deteksi Jenis Beras",
        "",
        f"Tanggal pembuatan: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. Tujuan",
        "",
        "Sistem ini dibuat untuk mendeteksi dan membedakan beberapa kelas beras: "
        + ", ".join(class_names or DEFAULT_CLASSES)
        + ".",
        "",
        "## 2. Dataset",
        "",
        "Dataset awal berupa folder per kelas. Karena tidak tersedia anotasi bounding box manual, label YOLO dibuat otomatis dengan satu kotak penuh pada setiap gambar. Pendekatan ini sesuai untuk foto satu objek/produk per frame, tetapi perlu anotasi manual bila gambar berisi banyak objek atau objek tidak memenuhi frame.",
        "",
        "| Split | Jumlah | Rincian kelas |",
        "| --- | ---: | --- |",
    ]

    for split in ["train", "val", "test"]:
        total = sum(counts[split].values())
        detail = ", ".join(f"{name}: {counts[split][name]}" for name in class_names)
        lines.append(f"| {split} | {total} | {detail} |")

    lines.extend(
        [
            "",
            "## 3. Metodologi",
            "",
            "1. Mengumpulkan gambar dari folder kelas.",
            "2. Mengubah struktur data menjadi format YOLO: `images/{train,val,test}` dan `labels/{train,val,test}`.",
            "3. Membuat label `.txt` YOLO dengan format `class_id x_center y_center width height`.",
            "4. Melatih model YOLOv8 menggunakan transfer learning dari bobot pretrained.",
            "5. Mengevaluasi model dengan precision, recall, mAP50, dan mAP50-95.",
            "",
            "## 4. Eksperimen",
            "",
            "Model default: `yolov8n.pt`, ukuran gambar 416, batch 8. Jumlah epoch dapat dinaikkan untuk hasil final yang lebih stabil.",
            "",
            "## 5. Hasil",
            "",
        ]
    )

    if run_dir and (run_dir / "results.csv").exists():
        lines.append(f"Hasil training tersimpan di `{run_dir}`. File metrik utama: `{run_dir / 'results.csv'}`.")
    else:
        lines.append("Jalankan training untuk menghasilkan metrik akhir dan bobot model `best.pt`.")

    lines.extend(
        [
            "",
            "## 6. Analisis",
            "",
            "Dataset sudah cukup untuk baseline, tetapi anotasi kotak penuh membuat evaluasi deteksi cenderung mengukur klasifikasi kelas dan lokalisasi kasar. Untuk laporan final yang lebih kuat, lakukan pelabelan manual bounding box pada sebagian atau seluruh gambar menggunakan LabelImg, CVAT, atau Roboflow.",
            "",
            "## 7. Cara Menjalankan",
            "",
            "```bash",
            "python be.py prepare",
            "python be.py train --epochs 30",
            "python be.py evaluate --model runs/rice_detection/weights/best.pt --split test",
            "python be.py predict --model runs/rice_detection/weights/best.pt --source dataset/IR42",
            "```",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Laporan ditulis: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline deteksi jenis beras berbasis YOLO.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Buat dataset YOLO dari folder kelas.")
    prepare.add_argument("--source", type=Path, default=Path("dataset"))
    prepare.add_argument("--output", type=Path, default=Path("yolo_rice_dataset"))
    prepare.add_argument("--train-ratio", type=float, default=0.7)
    prepare.add_argument("--val-ratio", type=float, default=0.2)
    prepare.add_argument("--seed", type=int, default=42)

    train = subparsers.add_parser("train", help="Latih model YOLO.")
    train.add_argument("--data", type=Path, default=Path("yolo_rice_dataset/data.yaml"))
    train.add_argument("--model", default="yolov8n.pt")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--imgsz", type=int, default=416)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--project", type=Path, default=Path("runs"))
    train.add_argument("--name", default="rice_detection")

    evaluate = subparsers.add_parser("evaluate", help="Evaluasi model YOLO.")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, default=Path("yolo_rice_dataset/data.yaml"))
    evaluate.add_argument("--imgsz", type=int, default=416)
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="test")

    pred = subparsers.add_parser("predict", help="Prediksi gambar/folder baru.")
    pred.add_argument("--model", type=Path, required=True)
    pred.add_argument("--source", type=Path, required=True)
    pred.add_argument("--imgsz", type=int, default=416)
    pred.add_argument("--conf", type=float, default=0.25)
    pred.add_argument("--output", type=Path, default=Path("runs"))

    report = subparsers.add_parser("report", help="Buat laporan akhir markdown.")
    report.add_argument("--output", type=Path, default=Path("laporan_akhir.md"))
    report.add_argument("--dataset", type=Path, default=Path("yolo_rice_dataset"))
    report.add_argument("--run-dir", type=Path, default=None)

    all_cmd = subparsers.add_parser("all", help="Prepare, train, evaluate, dan buat laporan.")
    all_cmd.add_argument("--epochs", type=int, default=30)
    all_cmd.add_argument("--imgsz", type=int, default=416)
    all_cmd.add_argument("--batch", type=int, default=8)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "prepare":
        prepare_dataset(args.source, args.output, args.train_ratio, args.val_ratio, args.seed)
    elif args.command == "train":
        train_model(args.data, args.model, args.epochs, args.imgsz, args.batch, args.project, args.name)
    elif args.command == "evaluate":
        evaluate_model(args.model, args.data, args.imgsz, args.split)
    elif args.command == "predict":
        predict(args.model, args.source, args.imgsz, args.conf, args.output)
    elif args.command == "report":
        write_report(args.output, args.dataset, args.run_dir)
    elif args.command == "all":
        data_yaml = prepare_dataset(Path("dataset"), Path("yolo_rice_dataset"), 0.7, 0.2, 42)
        best_model = train_model(
            data_yaml,
            "yolov8n.pt",
            args.epochs,
            args.imgsz,
            args.batch,
            Path("runs"),
            "rice_detection",
        )
        evaluate_model(best_model, data_yaml, args.imgsz, "test")
        write_report(Path("laporan_akhir.md"), Path("yolo_rice_dataset"), best_model.parents[1])


if __name__ == "__main__":
    main()
