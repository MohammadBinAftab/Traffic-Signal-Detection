"""
Traffic Sign Detection - Streamlit App
Uses YOLOv8 for detection, OpenCV for video handling.

Install:  pip install streamlit ultralytics opencv-python Pillow
Run:      streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile, os, time
from pathlib import Path

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Sign Detector",
    page_icon="🚦",
    layout="wide"
)

# ─── Class Names (43 GTSRB classes) ───────────────────────────────────────────
CLASS_NAMES = [
    'Speed limit 20','Speed limit 30','Speed limit 50','Speed limit 60',
    'Speed limit 70','Speed limit 80','End speed limit 80','Speed limit 100',
    'Speed limit 120','No passing','No passing >3.5t','Right-of-way junction',
    'Priority road','Yield','Stop','No vehicles','No trucks','No entry',
    'General caution','Dangerous curve left','Dangerous curve right',
    'Double curve','Bumpy road','Slippery road','Road narrows right',
    'Road work','Traffic signals','Pedestrians','Children crossing',
    'Bicycles crossing','Beware ice/snow','Wild animals','End restrictions',
    'Turn right ahead','Turn left ahead','Ahead only','Go straight or right',
    'Go straight or left','Keep right','Keep left','Roundabout mandatory',
    'End no passing','End no passing >3.5t'
]

# Colour per class (BGR for OpenCV, then converted to RGB for display)
COLORS = np.random.default_rng(42).integers(50, 220, size=(43, 3)).tolist()

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)

# ─── Inference on single frame ────────────────────────────────────────────────
def detect_frame(model, frame: np.ndarray, conf_thresh: float, img_size: int):
    """
    Run YOLO detection on a BGR frame (OpenCV format).
    Returns annotated RGB image + list of detections.
    """
    results = model(frame, imgsz=img_size, conf=conf_thresh, verbose=False)[0]
    detections = []

    for box in results.boxes:
        cls_id  = int(box.cls[0])
        conf    = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        label   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'Class {cls_id}'
        color   = COLORS[cls_id % len(COLORS)]

        # Draw bounding box (OpenCV expects BGR)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        text = f'{label}: {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(frame, text, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        detections.append({'label': label, 'confidence': f'{conf:.2%}',
                            'bbox': f'({x1},{y1}) → ({x2},{y2})'})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb, detections

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
model_file = st.sidebar.file_uploader(
    "Upload your trained best.pt", type=["pt"],
    help="Upload the best.pt file downloaded from Kaggle"
)
conf_thresh = st.sidebar.slider("Confidence threshold", 0.1, 0.95, 0.40, 0.05)
img_size    = st.sidebar.select_slider("Inference image size", [320, 416, 640], value=416)

model = None
if model_file:
    # Save uploaded model to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
        tmp.write(model_file.read())
        tmp_path = tmp.name
    try:
        model = load_model(tmp_path)
        st.sidebar.success("✅ Model loaded!")
    except Exception as e:
        st.sidebar.error(f"Model error: {e}")
else:
    st.sidebar.info("Upload your best.pt to begin.")

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.title("🚦 Traffic Sign Detector")
st.caption("YOLOv8 · GTSRB · 43 Classes")

tab1, tab2, tab3 = st.tabs(["📷 Image", "🎬 Video File", "📹 Webcam"])

# ══ TAB 1: Image ══════════════════════════════════════════════════════════════
with tab1:
    uploaded_img = st.file_uploader("Upload an image", type=["jpg","jpeg","png","ppm"],
                                     key="img_up")
    if uploaded_img and model:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Could not decode image.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader("Detected")
                annotated, dets = detect_frame(model, frame.copy(), conf_thresh, img_size)
                st.image(annotated, use_container_width=True)
            if dets:
                st.subheader(f"Found {len(dets)} sign(s)")
                st.table(dets)
            else:
                st.info("No traffic signs detected above threshold.")
    elif uploaded_img and not model:
        st.warning("Please upload your model in the sidebar first.")

# ══ TAB 2: Video File ═════════════════════════════════════════════════════════
with tab2:
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"],
                                     key="vid_up")
    if uploaded_vid and model:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_vid.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)   # OpenCV handles video reading
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        step         = max(1, int(fps // 5))   # process ~5 fps for speed

        stframe = st.empty()
        progress = st.progress(0, text="Processing video…")
        det_log  = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()          # OpenCV reads frame
            if not ret:
                break
            if frame_idx % step == 0:
                annotated, dets = detect_frame(model, frame.copy(), conf_thresh, img_size)
                stframe.image(annotated, channels="RGB", use_container_width=True)
                det_log.extend(dets)
                progress.progress(
                    min(frame_idx / max(total_frames,1), 1.0),
                    text=f"Frame {frame_idx}/{total_frames}"
                )
            frame_idx += 1

        cap.release()
        os.unlink(tfile.name)
        progress.empty()
        st.success("✅ Video processed!")

        if det_log:
            # Summarise unique signs
            from collections import Counter
            counts = Counter(d['label'] for d in det_log)
            st.subheader("Signs detected in video")
            st.bar_chart(counts)
    elif uploaded_vid and not model:
        st.warning("Please upload your model in the sidebar first.")

# ══ TAB 3: Webcam ═════════════════════════════════════════════════════════════
with tab3:
    st.info("Uses your default webcam (index 0). Press **Stop** to end.")
    run_cam = st.toggle("Start Webcam")

    if run_cam and model:
        cap = cv2.VideoCapture(0)            # OpenCV opens webcam
        if not cap.isOpened():
            st.error("Cannot access webcam.")
        else:
            stframe  = st.empty()
            stop_btn = st.button("⏹ Stop Webcam")
            det_area = st.empty()

            while not stop_btn:
                ret, frame = cap.read()      # OpenCV reads each frame
                if not ret:
                    st.warning("Frame read failed.")
                    break
                annotated, dets = detect_frame(model, frame.copy(), conf_thresh, img_size)
                stframe.image(annotated, channels="RGB", use_container_width=True)
                if dets:
                    det_area.table(dets)
                time.sleep(0.03)             # ~30 fps cap
            cap.release()
    elif run_cam and not model:
        st.warning("Please upload your model in the sidebar first.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Final Year Project · Traffic Sign Detection · YOLOv8 + OpenCV + Streamlit")
