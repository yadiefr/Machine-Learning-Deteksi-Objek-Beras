from __future__ import annotations

import hashlib
import io
from collections import Counter
from pathlib import Path

import streamlit as st
from PIL import Image

from gui_predict import draw_detections, tighten_box


DEFAULT_MODEL = Path("runs/detect/runs/rice_detection/weights/best.pt")


st.set_page_config(
    page_title="Deteksi Jenis Beras",
    page_icon="🌾",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { --ink: #eaf2ff; --muted: #91a4bf; --lime: #c7f464; }
    .stApp {
        background:
            radial-gradient(circle at 85% 5%, rgba(45, 212, 191, .13), transparent 27rem),
            radial-gradient(circle at 10% 80%, rgba(132, 204, 22, .08), transparent 24rem),
            #0a1020;
        color: var(--ink);
    }
    .block-container {
        max-width: 1320px;
        padding-top: 1.1rem;
        padding-bottom: 2rem;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMain"] h1, [data-testid="stMain"] h2,
    [data-testid="stMain"] h3, [data-testid="stMain"] p,
    [data-testid="stMain"] label, [data-testid="stMain"] small {
        color: var(--ink);
    }
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p { color: var(--muted); }
    .topbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: .65rem 1rem; margin-bottom: 1rem;
        border: 1px solid rgba(148, 163, 184, .18); border-radius: 14px;
        background: rgba(15, 23, 42, .72); backdrop-filter: blur(12px);
    }
    .brand { font-weight: 800; letter-spacing: -.02em; color: white; }
    .brand-mark {
        display: inline-grid; place-items: center; width: 34px; height: 34px;
        margin-right: .65rem; border-radius: 10px; background: var(--lime);
    }
    .status-pill {
        padding: .35rem .7rem; border-radius: 999px; font-size: .78rem;
        color: #d9f99d; background: rgba(132, 204, 22, .12);
        border: 1px solid rgba(190, 242, 100, .22);
    }
    .hero {
        padding: 1.25rem 0 1rem; margin-bottom: .2rem;
    }
    .hero .eyebrow {
        color: var(--lime); font-size: .72rem; font-weight: 800;
        letter-spacing: .14em; text-transform: uppercase;
    }
    .hero h1 {
        margin: .25rem 0; color: white; font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 1.02; letter-spacing: -.055em;
    }
    .hero p { margin: 0; color: var(--muted); max-width: 680px; font-size: 1rem; }
    [data-testid="stExpander"] {
        background: rgba(15, 23, 42, .78); border-color: rgba(148, 163, 184, .2);
        border-radius: 14px;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary * {
        color: #000000 !important;
    }
    [data-testid="stFileUploader"] section {
        background: rgba(15, 23, 42, .7); border: 1px dashed #52647e;
        border-radius: 14px;
    }
    [data-testid="stFileUploader"] section:hover { border-color: var(--lime); }
    [data-testid="stFileUploader"] section div,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small {
        color: #cbd5e1 !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploader"] section button {
        color: #101827 !important;
        background: #f8fafc !important;
        border-color: #f8fafc !important;
        font-weight: 750;
    }
    [data-testid="stFileUploaderFile"] {
        color: #f8fafc !important;
        background: rgba(30, 41, 59, .9);
    }
    [data-testid="stFileUploaderFile"] * { color: #f8fafc !important; }
    [data-testid="stTextInput"] input {
        color: #f8fafc !important;
        background: #111b2e !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--lime); color: #000000 !important; border: 0; font-weight: 800;
        box-shadow: 0 8px 24px rgba(190, 242, 100, .18);
    }
    .stButton > button[kind="primary"]:hover { background: #A1D64D; color: #000000 !important; }
    .stButton > button[kind="primary"] * { color: #000000 !important; }
    [data-testid="stDownloadButton"] > button {
        width: 100%;
        background: var(--lime) !important;
        color: #17220a !important;
        border: 0 !important;
        font-weight: 800;
        box-shadow: 0 8px 24px rgba(190, 242, 100, .18);
    }
    [data-testid="stDownloadButton"] > button:hover {
        background: #d9ff83 !important;
        color: #17220a !important;
    }
    [data-testid="stDownloadButton"] > button * { color: #17220a !important; }
    .panel-label {
        display: flex; align-items: center; gap: .55rem; margin: .7rem 0 .45rem;
        color: white; font-weight: 750; font-size: 1rem;
    }
    .panel-number {
        display: inline-grid; place-items: center; width: 24px; height: 24px;
        border-radius: 7px; background: rgba(199, 244, 100, .14); color: var(--lime);
        font-size: .75rem;
    }
    div[data-testid="stImage"] img {
        width: 100% !important;
        height: min(43vh, 420px) !important;
        object-fit: contain !important;
        background: #060b15;
        border: 1px solid #233047;
        border-radius: 14px;
    }
    [data-testid="stMetric"] {
        padding: .8rem 1rem; border-radius: 13px;
        background: rgba(15, 23, 42, .78); border: 1px solid rgba(148, 163, 184, .18);
    }
    [data-testid="stMetricValue"] { color: var(--lime); }
    hr { border-color: rgba(148, 163, 184, .18) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def run_detection(
    image: Image.Image,
    model_path: Path,
    confidence: float,
    tight_box: bool,
) -> tuple[Image.Image, list[tuple[str, float, tuple[int, int, int, int]]]]:
    model = load_model(str(model_path.resolve()))
    results = model.predict(
        source=image,
        conf=confidence,
        imgsz=416,
        max_det=10,
        verbose=False,
    )

    result = results[0]
    detections: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        score = float(box.conf[0])
        xyxy = tuple(int(round(value)) for value in box.xyxy[0].tolist())
        if tight_box:
            xyxy = tighten_box(image, xyxy)
        detections.append((str(result.names[class_id]), score, xyxy))

    return draw_detections(image, detections), detections


def image_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


st.markdown(
    """
    <div class="topbar">
      <div class="brand"><span class="brand-mark">🌾</span>Deteksi jenis beras dengan YOLO</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("⚙️  Pengaturan model"):
    model_col, confidence_col, box_col = st.columns([2, 1.25, 1])
    with model_col:
        model_value = st.text_input("Lokasi model YOLO", value=str(DEFAULT_MODEL))
    with confidence_col:
        confidence = st.slider(
            "Confidence minimum",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Turunkan nilainya jika objek tidak terdeteksi.",
        )
    with box_col:
        tight_box = st.toggle(
            "Kotak adaptif",
            value=True,
            help="Menyesuaikan kotak dengan area beras.",
        )

st.markdown(
    '<div class="panel-label"><span class="panel-number">01</span> Pilih gambar</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Unggah gambar beras",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Format yang didukung: JPG, PNG, BMP, dan WEBP.",
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info("Pilih sebuah gambar untuk memulai deteksi.", icon="📷")
    st.stop()

try:
    source_image = Image.open(uploaded_file).convert("RGB")
except Exception as exc:
    st.error(f"Gambar tidak dapat dibuka: {exc}")
    st.stop()

file_key = hashlib.sha1(uploaded_file.getvalue()).hexdigest()
if st.session_state.get("uploaded_file_key") != file_key:
    for key in ("result_image", "detections", "source_name"):
        st.session_state.pop(key, None)
    st.session_state["uploaded_file_key"] = file_key

action_col, info_col = st.columns([1, 2])
with action_col:
    detect_clicked = st.button("Deteksi Sekarang", type="primary", use_container_width=True)
with info_col:
    st.caption(f"Ukuran gambar: {source_image.width} × {source_image.height} px")

if detect_clicked:
    model_path = Path(model_value).expanduser()
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parent / model_path

    if not model_path.is_file():
        st.error(f"Model tidak ditemukan: `{model_path}`")
        st.stop()

    try:
        with st.spinner("Model sedang menganalisis gambar..."):
            result_image, detections = run_detection(
                source_image, model_path, confidence, tight_box
            )
    except Exception as exc:
        st.error(f"Prediksi gagal: {exc}")
        st.stop()

    st.session_state["result_image"] = result_image
    st.session_state["detections"] = detections
    st.session_state["source_name"] = Path(uploaded_file.name).stem

st.markdown(
    '<div class="panel-label"><span class="panel-number">02</span> Bandingkan hasil</div>',
    unsafe_allow_html=True,
)

image_col, result_col = st.columns(2, gap="large")
with image_col:
    st.caption("GAMBAR ASLI")
    st.image(source_image, use_container_width=True)

with result_col:
    st.caption("HASIL DETEKSI")
    if "result_image" in st.session_state:
        result_image = st.session_state["result_image"]
        st.image(result_image, use_container_width=True)
        st.download_button(
            "Simpan Gambar Hasil Deteksi",
            data=image_bytes(result_image),
            file_name=f"{st.session_state['source_name']}_hasil.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )
    else:
        st.info("Klik **Deteksi Sekarang** untuk menampilkan hasil di sini.", icon="🔍")

if "result_image" in st.session_state:
    detections = st.session_state["detections"]
    st.divider()
    st.markdown(
        '<div class="panel-label"><span class="panel-number">03</span> Ringkasan deteksi</div>',
        unsafe_allow_html=True,
    )
    if not detections:
        st.warning("Tidak ada objek terdeteksi. Coba turunkan confidence.")
    else:
        best_class, best_score, _ = max(detections, key=lambda item: item[1])
        counts = Counter(name for name, _score, _box in detections)
        metric_class, metric_score, metric_count = st.columns(3)
        metric_class.metric("Prediksi utama", best_class)
        metric_score.metric("Confidence tertinggi", f"{best_score:.1%}")
        metric_count.metric("Jumlah deteksi", len(detections))

        detail_col, recap_col = st.columns(2, gap="large")
        with detail_col:
            st.write("**Rincian deteksi**")
            for index, (class_name, score, _box) in enumerate(detections, start=1):
                st.write(f"{index}. {class_name} — {score:.1%}")

        with recap_col:
            st.write("**Rekap kelas**")
            for class_name, count in counts.most_common():
                st.write(f"• {class_name}: {count}")
