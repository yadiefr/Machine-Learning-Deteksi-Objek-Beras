# Laporan Akhir Deteksi Jenis Beras

Tanggal pembuatan: 2026-06-17 16:57

## 1. Tujuan

Sistem ini dibuat untuk mendeteksi dan membedakan beberapa kelas beras: Beras_Gundukan, Beras_Patahan, Beras_Utuh, IR42, IR64, Ketan, Pandan.

## 2. Dataset

Dataset awal berupa folder per kelas. Karena tidak tersedia anotasi bounding box manual, label YOLO dibuat otomatis dengan satu kotak penuh pada setiap gambar. Pendekatan ini sesuai untuk foto satu objek/produk per frame, tetapi perlu anotasi manual bila gambar berisi banyak objek atau objek tidak memenuhi frame.

| Split | Jumlah | Rincian kelas |
| --- | ---: | --- |
| train | 499 | Beras_Gundukan: 73, Beras_Patahan: 71, Beras_Utuh: 79, IR42: 80, IR64: 58, Ketan: 71, Pandan: 67 |
| val | 141 | Beras_Gundukan: 21, Beras_Patahan: 20, Beras_Utuh: 22, IR42: 23, IR64: 16, Ketan: 20, Pandan: 19 |
| test | 78 | Beras_Gundukan: 11, Beras_Patahan: 11, Beras_Utuh: 12, IR42: 12, IR64: 10, Ketan: 11, Pandan: 11 |

## 3. Metodologi

1. Mengumpulkan gambar dari folder kelas.
2. Mengubah struktur data menjadi format YOLO: `images/{train,val,test}` dan `labels/{train,val,test}`.
3. Membuat label `.txt` YOLO dengan format `class_id x_center y_center width height`.
4. Melatih model YOLOv8 menggunakan transfer learning dari bobot pretrained.
5. Mengevaluasi model dengan precision, recall, mAP50, dan mAP50-95.

## 4. Eksperimen

Model default: `yolov8n.pt`, ukuran gambar 416, batch 8. Jumlah epoch dapat dinaikkan untuk hasil final yang lebih stabil.

## 5. Hasil

Hasil training tersimpan di `runs\detect\runs\rice_detection`. File metrik utama: `runs\detect\runs\rice_detection\results.csv`.

## 6. Analisis

Dataset sudah cukup untuk baseline, tetapi anotasi kotak penuh membuat evaluasi deteksi cenderung mengukur klasifikasi kelas dan lokalisasi kasar. Untuk laporan final yang lebih kuat, lakukan pelabelan manual bounding box pada sebagian atau seluruh gambar menggunakan LabelImg, CVAT, atau Roboflow.

## 7. Cara Menjalankan

```bash
python run.py prepare
python run.py train --epochs 30
python run.py evaluate --model runs/rice_detection/weights/best.pt --split test
python run.py predict --model runs/rice_detection/weights/best.pt --source dataset/IR42
```
