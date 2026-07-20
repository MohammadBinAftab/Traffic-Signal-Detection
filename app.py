"""
Traffic Sign Detection - Streamlit App
Uses YOLOv8 for detection, OpenCV for video handling.
RL agent (Q-Learning) adaptively controls the confidence threshold.

Install:  pip install streamlit ultralytics opencv-python Pillow numpy
Run:      python -m streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile, os, time
from rl_agent import AdaptiveThresholdAgent

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

# Unique colour per class
COLORS = np.random.default_rng(42).integers(50, 220, size=(43, 3)).tolist()

# ─── Load Model (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)

# ─── Load RL Agent (cached) ───────────────────────────────────────────────────
@st.cache_resource
def get_agent():
    return AdaptiveThresholdAgent(q_table_path="q_table.json")

agent = get_agent()

# ─── Frame counter in session state ───────────────────────────────────────────
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

# ─── Inference on single frame ────────────────────────────────────────────────
def detect_frame(model, frame: np.ndarray, conf_thresh: float, img_size):
    """
    Run YOLO detection on a BGR frame (OpenCV format).
    Supports either an integer size (e.g. 224) or the string "Multi-Scale (Robust)".
    Returns annotated RGB image + list of detections.
    """
    if img_size == "Multi-Scale (Robust)":
        scales = [160, 224, 416, 640]
    else:
        scales = [int(img_size)]

    all_boxes = []
    all_scores = []
    all_class_ids = []

    for sz in scales:
        results = model(frame, imgsz=sz, conf=conf_thresh, verbose=False)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w = x2 - x1
            h = y2 - y1
            all_boxes.append([x1, y1, w, h])
            all_scores.append(conf)
            all_class_ids.append(cls_id)

    # Run Non-Maximum Suppression to deduplicate boxes across different scales
    indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, conf_thresh, 0.45)
    
    detections = []
    if len(indices) > 0:
        for idx in indices.flatten():
            x, y, w, h = all_boxes[idx]
            cls_id = all_class_ids[idx]
            conf = all_scores[idx]
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'Class {cls_id}'
            color = COLORS[cls_id % len(COLORS)]

            # Draw bounding box — OpenCV uses BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f'{label}: {conf:.2f}'
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
            cv2.putText(frame, text, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            detections.append({
                'label': label,
                'confidence': f'{conf:.2%}',
                'bbox': f'({x1},{y1}) → ({x2},{y2})'
            })

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb, detections


def rl_detect(model, frame: np.ndarray, img_size: int):
    """
    RL-assisted detection:
      1. Agent observes frame → picks confidence threshold
      2. YOLO runs with that threshold
      3. Agent receives reward and updates Q-table
      4. Q-table saved every 50 frames
    Returns annotated RGB image, detections, and the threshold used.
    """
    # 1 — Agent picks threshold based on frame conditions
    rl_threshold = agent.get_threshold(frame)

    # 2 — YOLO runs with agent's threshold
    annotated, dets = detect_frame(model, frame.copy(), rl_threshold, img_size)

    # 3 — Agent learns from result
    agent.update(frame, len(dets))

    # 4 — Save Q-table every 50 frames so learning persists between sessions
    st.session_state.frame_count += 1
    if st.session_state.frame_count % 50 == 0:
        agent.save()

    return annotated, dets, rl_threshold


# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

model_file = st.sidebar.file_uploader(
    "Upload your trained best.pt", type=["pt"],
    help="Upload the best.pt file downloaded from Kaggle"
)

# Mode toggle — manual slider OR RL agent
use_rl  = st.sidebar.toggle("🤖 Use RL Adaptive Threshold", value=True,
                              help="Let the Q-Learning agent choose the threshold automatically")

# Manual slider shown only when RL is OFF
if not use_rl:
    conf_thresh = st.sidebar.slider("Confidence threshold", 0.1, 0.95, 0.40, 0.05)
else:
    conf_thresh = 0.40   # fallback default, overridden by agent each frame

img_size = st.sidebar.selectbox("Inference image size / Mode", ["Multi-Scale (Robust)", 96, 128, 160, 224, 320, 416, 640], index=0)

# ── RL Agent status panel ─────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 RL Agent Status")

if use_rl:
    if agent.last_action is not None:
        current_threshold = agent.THRESHOLD_OPTIONS[agent.last_action]
        st.sidebar.metric("Current Threshold", f"{current_threshold:.2f}")
    else:
        st.sidebar.metric("Current Threshold", "Waiting…")

    st.sidebar.metric("States Learned",  len(agent.q_table))
    st.sidebar.metric("Frames Processed", st.session_state.frame_count)

    if st.sidebar.button("💾 Save Q-Table Now"):
        agent.save()
        st.sidebar.success("Q-table saved!")

    st.sidebar.caption(
        "Agent adapts threshold based on frame brightness, "
        "blur level, and recent detection count."
    )
else:
    st.sidebar.caption("RL agent is OFF. Using manual slider above.")

# ── Model loading ─────────────────────────────────────────────────────────────
model = None
DEFAULT_MODEL_PATH = "best.pt"

if model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
        tmp.write(model_file.read())
        tmp_path = tmp.name
    try:
        model = load_model(tmp_path)
        st.sidebar.success("✅ Model loaded from upload!")
    except Exception as e:
        st.sidebar.error(f"Model error: {e}")
elif os.path.exists(DEFAULT_MODEL_PATH):
    try:
        model = load_model(DEFAULT_MODEL_PATH)
        st.sidebar.success("✅ Auto-loaded local 'best.pt'!")
    except Exception as e:
        st.sidebar.error(f"Error loading local model: {e}")
else:
    st.sidebar.info("Upload your best.pt or place it in the project folder to begin.")

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.title("🚦 Traffic Sign Detector")
st.caption("YOLOv8 · GTSRB · 43 Classes · Q-Learning Adaptive Threshold")

tab1, tab2, tab3 = st.tabs(["📷 Image", "🎬 Video File", "📹 Webcam"])

# ══ TAB 1: Image ══════════════════════════════════════════════════════════════
with tab1:
    uploaded_img = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "ppm"], key="img_up"
    )

    if uploaded_img and model:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("Could not decode image.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col2:
                st.subheader("Detected")
                if use_rl:
                    annotated, dets, used_thresh = rl_detect(model, frame, img_size)
                    st.caption(f"🤖 RL agent used threshold: **{used_thresh:.2f}**")
                else:
                    annotated, dets = detect_frame(model, frame.copy(), conf_thresh, img_size)
                    st.caption(f"Manual threshold: **{conf_thresh:.2f}**")

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
    uploaded_vid = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid_up"
    )

    if uploaded_vid and model:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_vid.read())
        tfile.close()

        cap          = cv2.VideoCapture(tfile.name)   # OpenCV handles video reading
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        step         = max(1, int(fps // 5))          # process ~5 fps for speed

        stframe      = st.empty()
        thresh_info  = st.empty()
        progress     = st.progress(0, text="Processing video…")
        det_log      = []
        frame_idx    = 0

        while True:
            ret, frame = cap.read()       # OpenCV reads each frame
            if not ret:
                break

            if frame_idx % step == 0:
                if use_rl:
                    annotated, dets, used_thresh = rl_detect(model, frame, img_size)
                    thresh_info.caption(f"🤖 RL threshold this frame: **{used_thresh:.2f}**")
                else:
                    annotated, dets = detect_frame(model, frame.copy(), conf_thresh, img_size)

                stframe.image(annotated, channels="RGB", use_container_width=True)
                det_log.extend(dets)
                progress.progress(
                    min(frame_idx / max(total_frames, 1), 1.0),
                    text=f"Frame {frame_idx}/{total_frames}"
                )

            frame_idx += 1

        cap.release()
        os.unlink(tfile.name)
        progress.empty()
        thresh_info.empty()
        st.success("✅ Video processed!")

        if det_log:
            from collections import Counter
            counts = Counter(d['label'] for d in det_log)
            st.subheader("Signs detected in video")
            st.bar_chart(counts)

    elif uploaded_vid and not model:
        st.warning("Please upload your model in the sidebar first.")

# ══ TAB 3: Webcam ═════════════════════════════════════════════════════════════
with tab3:
    webcam_mode = st.radio("Select Webcam Mode", ["📸 Camera Snapshot (Works Online & Local)", "📹 Live Stream (Works Local Only)"])

    if webcam_mode == "📸 Camera Snapshot (Works Online & Local)":
        uploaded_snap = st.camera_input("Take a photo to detect traffic signs")
        if uploaded_snap and model:
            file_bytes = np.frombuffer(uploaded_snap.read(), np.uint8)
            frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is None:
                st.error("Could not decode image.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Photo")
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                with col2:
                    st.subheader("Detected Signs")
                    if use_rl:
                        annotated, dets, used_thresh = rl_detect(model, frame, img_size)
                        st.caption(f"🤖 RL agent used threshold: **{used_thresh:.2f}**")
                    else:
                        annotated, dets = detect_frame(model, frame.copy(), conf_thresh, img_size)
                        st.caption(f"Manual threshold: **{conf_thresh:.2f}**")

                    st.image(annotated, use_container_width=True)

                if dets:
                    st.subheader(f"Found {len(dets)} sign(s)")
                    st.table(dets)
                else:
                    st.info("No traffic signs detected above threshold.")

        elif uploaded_snap and not model:
            st.warning("Please upload your model in the sidebar first.")

    else:
        st.info("Uses your default webcam (index 0). Toggle off to stop.")
        run_cam = st.toggle("Start Webcam")

        if run_cam and model:
            cap = cv2.VideoCapture(0)    # OpenCV opens webcam
            if not cap.isOpened():
                st.error("Cannot access webcam. Ensure you are running this app locally on your laptop.")
            else:
                stframe     = st.empty()
                thresh_disp = st.empty()
                stop_btn    = st.button("⏹ Stop Webcam")
                det_area    = st.empty()

                while not stop_btn:
                    ret, frame = cap.read()     # OpenCV reads each frame
                    if not ret:
                        st.warning("Frame read failed.")
                        break

                    if use_rl:
                        annotated, dets, used_thresh = rl_detect(model, frame, img_size)
                        thresh_disp.caption(f"🤖 RL threshold: **{used_thresh:.2f}**")
                    else:
                        annotated, dets = detect_frame(model, frame.copy(), conf_thresh, img_size)

                    stframe.image(annotated, channels="RGB", use_container_width=True)

                    if dets:
                        det_area.table(dets)

                    time.sleep(0.03)    # ~30 fps cap

                cap.release()

        elif run_cam and not model:
            st.warning("Please upload your model in the sidebar first.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Final Year Project · Traffic Sign Detection · YOLOv8 + OpenCV + Streamlit + Q-Learning")
