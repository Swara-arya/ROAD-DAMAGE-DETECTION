"""
RoadScan AI - Road Damage Detection dashboard (Streamlit)
Run:  streamlit run app.py
Put your trained .pt files in the 'models' folder next to this file.
"""
import glob
import hashlib
import io
import os
import tempfile
import time
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

try:
    import torch
    from torchvision.ops import batched_nms
    HAVE_NMS = True
except Exception:  # torchvision missing: ensemble falls back to plain merging
    HAVE_NMS = False

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DEFAULT_MODEL = "yolov8s_final.pt"

# name -> label, RDD code, colour (hex), severity weight
CLASS_INFO = {
    "longitudinal_crack": ("Longitudinal crack", "D00", "#3B82F6", 1),
    "transverse_crack":   ("Transverse crack",   "D10", "#A855F7", 1),
    "alligator_crack":    ("Alligator crack",    "D20", "#F59E0B", 2),
    "pothole":            ("Pothole",            "D40", "#EF4444", 3),
}

# Results measured in Colab on the India test set (1,166 images)
PERFORMANCE = pd.DataFrame([
    ["YOLOv8s final (+ pothole photos)", 0.378, 0.173, 0.417, 0.439, 22.5],
    ["YOLOv8s all countries -> India",   0.365, 0.173, 0.418, 0.427, 22.5],
    ["YOLOv8s all countries",            0.323, 0.153, 0.356, 0.326, 22.5],
    ["YOLOv8s India only (baseline)",    0.316, 0.139, 0.363, 0.334, 22.5],
    ["YOLOv8n India only",               0.306, 0.143, 0.574, 0.348, 6.3],
    ["YOLOv8m India only",               0.303, 0.138, 0.599, 0.315, 52.0],
], columns=["Model", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "Size (MB)"])

PER_CLASS = pd.DataFrame({
    "Baseline": [0.268, 0.002, 0.636, 0.360],
    "Final model": [0.371, 0.078, 0.651, 0.413],
}, index=["Longitudinal crack", "Transverse crack", "Alligator crack", "Pothole"])


def info(name):
    return CLASS_INFO.get(name, (name.replace("_", " ").title(), "-", "#22C55E", 1))


def keep_mask(scores, classes, names, pothole_conf, crack_conf):
    """Separate thresholds: the main slider for potholes, a lower one for cracks."""
    thr = np.array([pothole_conf if info(names[int(c)])[1] == "D40" else crack_conf
                    for c in classes], dtype=float)
    return scores >= thr if len(scores) else np.zeros(0, dtype=bool)


def hex_to_bgr(h):
    return (int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16))


# ---------------------------------------------------------------- page setup
st.set_page_config(page_title="RoadScan AI", page_icon="🛣️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {padding: 1.4rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;
       background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 60%, #c2410c 140%);}
.hero h1 {color: #ffffff; margin: 0; font-size: 2.1rem; letter-spacing: .5px;}
.hero p {color: #dbe4f0; margin: .35rem 0 0 0; font-size: 1rem;}
.badge {display: inline-block; padding: .35rem .9rem; border-radius: 999px;
        color: #ffffff; font-weight: 700; font-size: .95rem;}
.chip {display: inline-block; width: 12px; height: 12px; border-radius: 3px;
       margin-right: 8px; vertical-align: middle;}
.small {color: #94a3b8; font-size: .85rem;}
div[data-testid="stMetric"] {background: rgba(148,163,184,.08); border: 1px solid rgba(148,163,184,.22);
                              border-radius: 12px; padding: .7rem 1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🛣️ RoadScan AI</h1>
  <p>Deep-learning road damage detection · YOLOv8 trained on RDD2022 (India + 5 countries) and real-world pothole photos</p>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = {}
if "pred_cache" not in st.session_state:
    st.session_state.pred_cache = {}


# ---------------------------------------------------------------- models
@st.cache_resource
def load_model(path):
    m = YOLO(path)
    m.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)  # warm-up so the first photo is fast
    return m


def run_models(models, img_bgr, imgsz, tta, iou):
    """Predict with one or several models; several = ensemble merged with class-wise NMS."""
    boxes, scores, classes = [], [], []
    for m in models:
        r = m.predict(img_bgr, conf=0.01, iou=iou, imgsz=imgsz, augment=tta, verbose=False)[0]
        if len(r.boxes):
            boxes.append(r.boxes.xyxy.cpu().numpy())
            scores.append(r.boxes.conf.cpu().numpy())
            classes.append(r.boxes.cls.cpu().numpy().astype(int))
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    b, s, c = np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)
    if len(models) > 1 and HAVE_NMS:
        keep = batched_nms(torch.tensor(b), torch.tensor(s), torch.tensor(c), iou).numpy()
        b, s, c = b[keep], s[keep], c[keep]
    order = np.argsort(-s)
    return b[order], s[order], c[order]


def draw(img_bgr, boxes, scores, classes, names):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    t = max(2, round(min(h, w) / 320))
    fs = max(0.45, min(h, w) / 1000)
    for (x1, y1, x2, y2), s, c in zip(boxes.astype(int), scores, classes):
        label, code, color, _ = info(names[int(c)])
        col = hex_to_bgr(color)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, t)
        text = f"{label} {s:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t - 1))
        ty = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 10
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty + base - 2), col, -1)
        cv2.putText(out, text, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, t - 1), cv2.LINE_AA)
    return out


def detections_table(boxes, scores, classes, names, shape):
    h, w = shape[:2]
    rows = []
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        label, code, _, weight = info(names[int(c)])
        rows.append({"Damage type": label, "Code": code, "Confidence": round(float(s), 3),
                     "Area (% of image)": round(100 * (x2 - x1) * (y2 - y1) / (w * h), 2),
                     "Severity weight": weight,
                     "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
    return pd.DataFrame(rows)


def assess(df):
    """Rule-based road condition from the detections (an estimate, not an engineering survey)."""
    if df.empty:
        return "Good", "#16A34A", "No visible damage detected", 0
    score = int(df["Severity weight"].sum())
    potholes = int((df["Code"] == "D40").sum())
    alligator = int((df["Code"] == "D20").sum())
    if score >= 8 or potholes >= 3:
        return "Very poor", "#DC2626", "Urgent repair recommended", score
    if potholes or alligator or len(df) >= 4:
        return "Poor", "#EA580C", "Schedule repair soon", score
    return "Fair", "#CA8A04", "Minor cracking - monitor", score


# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("## ⚙️ Detection settings")
model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")))
if not model_files:
    st.sidebar.error("No model found. Put your trained .pt file(s) in the 'models' folder.")
    st.stop()
all_names = [os.path.basename(m) for m in model_files]
default = [DEFAULT_MODEL] if DEFAULT_MODEL in all_names else all_names[:1]
chosen = st.sidebar.multiselect("Model(s)", all_names, default=default,
                                help="Pick two or more to run an ENSEMBLE: their boxes are combined, "
                                     "which usually finds more damage.")
if not chosen:
    st.sidebar.warning("Select at least one model.")
    st.stop()
models = [load_model(os.path.join(MODEL_DIR, n)) for n in chosen]
names = models[0].names

conf = st.sidebar.slider("Pothole threshold (D40)", 0.01, 0.90, 0.20, 0.01,
                         help="Only boxes the model is at least this sure about are shown.")
crack_conf = st.sidebar.slider("Crack threshold (D00, D10, D20)", 0.01, 0.90, 0.08, 0.01,
                               help="Cracks are thin and faint, so the model is less sure about them "
                                    "than about potholes. A lower threshold for cracks shows them "
                                    "without adding false pothole boxes.")
with st.sidebar.expander("Advanced"):
    iou = st.slider("Overlap (IoU) for merging boxes", 0.30, 0.90, 0.50, 0.05,
                    help="Boxes overlapping more than this are treated as the same damage.")
    imgsz = st.select_slider("Inference image size", options=[640, 960, 1280], value=640,
                             help="Larger can find thin cracks in big photos, but is slower.")
    tta = st.checkbox("Test-time augmentation (TTA)", value=False,
                      help="Also looks at flipped and rescaled copies of the image. Finds more, about 2-3x slower.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Damage classes**")
for key, (label, code, color, weight) in CLASS_INFO.items():
    st.sidebar.markdown(f"<span class='chip' style='background:{color}'></span>{label} ({code}) · severity {weight}",
                        unsafe_allow_html=True)
mode = "Ensemble of " + str(len(chosen)) if len(chosen) > 1 else chosen[0]
st.sidebar.markdown(f"<p class='small'>Active: {mode}{' + TTA' if tta else ''} · {imgsz}px</p>",
                    unsafe_allow_html=True)

tab_img, tab_vid, tab_report, tab_perf, tab_about = st.tabs(
    ["🔍 Image analysis", "🎥 Video analysis", "📋 Session report", "📈 Model performance", "ℹ️ About"])

# ---------------------------------------------------------------- image tab
with tab_img:
    files = st.file_uploader("Upload road photos (JPG / PNG)", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)
    if not files:
        st.info("Upload one or more road images to start. Dashcam-style photos give the best results.")
    for f in files or []:
        data = f.getvalue()
        key = (hashlib.md5(data).hexdigest(), tuple(chosen), imgsz, tta, iou)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if key not in st.session_state.pred_cache:
            t0 = time.perf_counter()
            preds = run_models(models, bgr, imgsz, tta, iou)
            st.session_state.pred_cache[key] = (preds, (time.perf_counter() - t0) * 1000)
        (b_all, s_all, c_all), ms = st.session_state.pred_cache[key]
        keep = keep_mask(s_all, c_all, names, conf, crack_conf)
        b, s, c = b_all[keep], s_all[keep], c_all[keep]
        df = detections_table(b, s, c, names, bgr.shape)
        cond, color, advice, score = assess(df)

        with st.container(border=True):
            st.subheader(f"📷 {f.name}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Damages found", len(df))
            m2.metric("Severity score", score)
            m3.metric("Damaged area", f"{min(df['Area (% of image)'].sum(), 100):.1f}%" if len(df) else "0%")
            m4.metric("Inference time", f"{ms:.0f} ms")
            st.markdown(f"Road condition: <span class='badge' style='background:{color}'>{cond}</span> "
                        f"&nbsp; {advice}", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            col1.image(img, caption="Original")
            annotated = draw(bgr, b, s, c, names)
            col2.image(annotated, caption="Detected damage", channels="BGR")

            if df.empty:
                if len(s_all):
                    top = int(np.argmax(s_all))
                    st.info(f"Strongest candidate: {info(names[int(c_all[top])])[0]} at confidence "
                            f"{s_all[top]:.2f}. Lower the slider, try an ensemble, or turn on TTA in Advanced.")
                else:
                    st.info("The model sees no damage candidates in this image.")
            else:
                t1, t2 = st.columns([3, 2])
                t1.dataframe(df.drop(columns=["x1", "y1", "x2", "y2"]), hide_index=True)
                t2.bar_chart(df["Damage type"].value_counts())

            ok, png = cv2.imencode(".png", annotated)
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download annotated image", png.tobytes(),
                               file_name=f"detected_{os.path.splitext(f.name)[0]}.png", mime="image/png",
                               key=f"png-{key}-{conf}-{crack_conf}")
            d2.download_button("⬇️ Download detections (CSV)", df.to_csv(index=False).encode(),
                               file_name=f"detections_{os.path.splitext(f.name)[0]}.csv", mime="text/csv",
                               key=f"csv-{key}-{conf}-{crack_conf}")

        counts = Counter(df["Damage type"]) if len(df) else Counter()
        st.session_state.history[f.name] = {
            "Time": datetime.now().strftime("%H:%M:%S"), "Image": f.name, "Model": mode,
            "Damages": len(df), "Severity": score, "Condition": cond,
            **{lbl: counts.get(lbl, 0) for lbl, _, _, _ in CLASS_INFO.values()}}

# ---------------------------------------------------------------- video tab
with tab_vid:
    vf = st.file_uploader("Upload a road / dashcam video", type=["mp4", "avi", "mov", "mkv"])
    every = st.slider("Analyse every Nth frame", 1, 15, 5,
                      help="Higher = faster on a laptop without a GPU.")
    if vf is not None and st.button("▶️ Start video analysis", type="primary"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vf.name)[1])
        tmp.write(vf.read())
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        v1, v2, v3 = st.columns(3)
        p_frames, p_dets, p_pot = v1.empty(), v2.empty(), v3.empty()
        frame_box, progress = st.empty(), st.progress(0.0)
        counts, timeline, i, done = Counter(), [], 0, 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if i % every == 0:
                b_all, s_all, c_all = run_models(models, frame, imgsz, tta, iou)
                keep = keep_mask(s_all, c_all, names, conf, crack_conf)
                b, s, c = b_all[keep], s_all[keep], c_all[keep]
                for cid in c:
                    counts[info(names[int(cid)])[0]] += 1
                timeline.append({"Time (s)": round(i / fps, 1), "Detections": int(len(c))})
                frame_box.image(draw(frame, b, s, c, names), channels="BGR")
                done += 1
                p_frames.metric("Frames analysed", done)
                p_dets.metric("Total detections", sum(counts.values()))
                p_pot.metric("Potholes", counts.get("Pothole", 0))
            i += 1
            progress.progress(min(i / total, 1.0))
        cap.release()
        os.unlink(tmp.name)
        st.success(f"Finished: {done} frames analysed out of {i}.")
        if timeline:
            st.markdown("**Detections over the video**")
            st.line_chart(pd.DataFrame(timeline).set_index("Time (s)"))
        if counts:
            summary = pd.DataFrame([{"Damage type": k, "Detections (all frames)": v} for k, v in counts.most_common()])
            st.dataframe(summary, hide_index=True)
            st.download_button("⬇️ Download video timeline (CSV)", pd.DataFrame(timeline).to_csv(index=False).encode(),
                               file_name="video_timeline.csv", mime="text/csv")

# ---------------------------------------------------------------- session report
with tab_report:
    hist = pd.DataFrame(list(st.session_state.history.values()))
    if hist.empty:
        st.info("Analyse some images first. Every image you analyse in this session is listed here.")
    else:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Images analysed", len(hist))
        r2.metric("Total damages", int(hist["Damages"].sum()))
        r3.metric("Images needing repair", int(hist["Condition"].isin(["Poor", "Very poor"]).sum()))
        r4.metric("Average severity", f"{hist['Severity'].mean():.1f}")
        st.dataframe(hist, hide_index=True)
        labels = [lbl for lbl, _, _, _ in CLASS_INFO.values()]
        st.markdown("**Damage found across all images**")
        st.bar_chart(hist[labels].sum())
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download session report (CSV)", hist.to_csv(index=False).encode(),
                           file_name="road_damage_session_report.csv", mime="text/csv")
        if c2.button("🗑️ Clear report"):
            st.session_state.history = {}
            st.rerun()

# ---------------------------------------------------------------- model performance
with tab_perf:
    st.markdown("### How the models compare")
    st.caption("Measured in Google Colab on the RDD2022 India test set (1,166 images).")
    p1, p2, p3 = st.columns(3)
    p1.metric("Best mAP@0.5 (India)", "0.378", "+0.062 vs baseline")
    p2.metric("Pothole photos mAP@0.5", "0.766", "+0.763 after Experiment E")
    p3.metric("Fastest model", "YOLOv8n", "78 FPS on a T4 GPU")
    st.bar_chart(PERFORMANCE.set_index("Model")["mAP@0.5"])
    st.dataframe(PERFORMANCE, hide_index=True)
    st.markdown("**AP@0.5 per damage class: baseline vs final model**")
    st.bar_chart(PER_CLASS)
    st.markdown("""
- **A bigger model did not help:** YOLOv8m is about 2.3x the size of YOLOv8s but scored lower, so data was the limit.
- **Transfer learning helped:** pre-training on all six countries, then fine-tuning on India, beat India-only training.
- **Domain adaptation worked:** adding 665 everyday pothole photos raised pothole-photo mAP from 0.003 to 0.766
  without hurting the crack classes.
""")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown("""
### About this project
**Problem.** Road inspection in India is mostly manual, slow and costly, and damaged roads contribute to accidents.
This app detects road damage automatically from a photo or dashcam video.

**Model.** YOLOv8s, a one-stage convolutional object detector, trained in Google Colab (Tesla T4) with transfer learning:
COCO weights → all six RDD2022 countries → India → India + real-world pothole photos.

**Data.** RDD2022 (Arya et al., 2022), 47,420 road images from six countries, and the Pothole Detection photo set
(665 images).

**Road condition.** A rule-based estimate from the detections: each damage has a severity weight
(cracks 1, alligator cracks 2, potholes 3). It is a screening aid, not an engineering survey.

**Limitations.** Recall on Indian dashcam images is about 0.44, so some damage is missed; transverse cracks are rare in
the training data; close-up or stock photos of cracks can be missed. Ensemble mode and TTA in the sidebar help.

*ANN and Deep Learning — Major Project*
""")
