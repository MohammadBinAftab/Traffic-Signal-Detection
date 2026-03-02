---

# 🚦 Traffic Sign Detection System — Documentation

## 1. Introduction
This project builds a real-time traffic sign detection system using YOLOv8 for detection, OpenCV for video handling, and Streamlit for the desktop UI. It detects all 43 classes of the GTSRB dataset from images, video files, and live webcam.

---

## 2. System Architecture
```
TRAINING (Kaggle)
  Train.csv → PPM to JPG converter → YOLO labels → YOLOv8 training → best.pt
                                                                          ↓ download
INFERENCE (Local PC)
  Input → OpenCV reads frame → YOLOv8 detects → OpenCV draws boxes → Streamlit shows result
```

---

## 3. Dataset — GTSRB
| Property | Value |
|---|---|
| Training images | ~51,839 |
| Classes | 43 |
| Annotations | Train.csv (Roi.X1, Y1, X2, Y2, ClassId, Path) |
| Format | PNG/PPM → converted to JPEG |

**Folder structure:**
```
gtsrb-german-traffic-sign/
├── Train.csv       ← root-level CSV with all annotations
├── Train/
│   ├── 0/          ← Speed limit 20
│   ├── 1/          ← Speed limit 30
│   └── ... (43 folders)
└── Test/
```

---

## 4. Training Configuration
| Parameter | Default | Best Accuracy |
|---|---|---|
| Model | yolov8n | yolov8s |
| Image size | 416 | 640 |
| Epochs | 30 | 50–100 |
| Batch | 32 | 16 |
| Horizontal flip | 0.0 (OFF) | Always OFF |

**Kaggle steps:** New Notebook → Add GTSRB dataset → GPU T4 → Run all cells → Download `best.pt`

---

## 5. Streamlit App
**Install:** `pip install streamlit ultralytics opencv-python Pillow numpy`  
**Run:** `python -m streamlit run app.py`

| Tab | Function |
|---|---|
| Image | Upload photo → see bounding boxes + table |
| Video | Upload video → processed at ~5fps + bar chart |
| Webcam | Live detection from camera |

**Confidence threshold:** 0.40 default. Lower = more detections, higher = stricter.  
**Image size:** Set to 416 (match your training size).

---

## 6. Expected Results
| Config | mAP50 |
|---|---|
| yolov8n, 30ep, 416 | ~82–87% |
| yolov8s, 50ep, 640 | ~92–95% |
| yolov8m, 100ep, 640 | ~95–97% |

---

## 7. Troubleshooting
| Problem | Fix |
|---|---|
| streamlit not recognized | `python -m streamlit run app.py` |
| No images found in train | PPM conversion failed — check DATASET_PATH |
| 0 samples from CSV | Print `df['Path'].iloc[0]` to verify path format |
| CUDA out of memory | Reduce BATCH to 16 |
| Low accuracy | More epochs, use yolov8s, IMG_SIZE=640 |

---

It seems the file system is having an issue right now. Try refreshing and sending the message again — I'll regenerate a downloadable `.docx` version for you. Alternatively, you can copy the content above into a Word document or Google Docs right now.
