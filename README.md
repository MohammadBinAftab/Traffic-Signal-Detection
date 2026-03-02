# 🚦 Traffic Sign Detection System

A real-time traffic sign detection system using **YOLOv8**, **OpenCV**, and **Streamlit**.  
Detects and classifies **43 types** of German traffic signs from images, videos, and live webcam.

---

## 📁 Project Structure

```
├── kaggle_train_gtsrb.ipynb   # Train the model on Kaggle
├── app.py                     # Streamlit desktop application
├── requirements.txt           # Python dependencies
└── best.pt                    # Your trained model (download from Kaggle)
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/traffic-sign-detection.git
cd traffic-sign-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your trained model
Download `best.pt` from your Kaggle training run and place it in the project folder.

### 4. Run the app
```bash
python -m streamlit run app.py
```
Open your browser at **http://localhost:8501**

---

## 🏋️ Training the Model (Kaggle)

1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook
2. Add the dataset: **gtsrb-german-traffic-sign**
3. Enable **GPU T4** under Settings → Accelerator
4. Upload and run `kaggle_train_gtsrb.ipynb`
5. Download `runs/gtsrb/weights/best.pt` from the Output panel

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Name | GTSRB — German Traffic Sign Recognition Benchmark |
| Training Images | ~51,839 |
| Classes | 43 |
| Source | [Kaggle GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) |

---

## 🎯 Model Performance

| Configuration | Expected mAP50 |
|--------------|----------------|
| YOLOv8n, 30 epochs, img=416 | ~82–87% |
| YOLOv8s, 50 epochs, img=640 | ~92–95% |

---

## 🖥️ App Features

- **Image Mode** — Upload a photo and detect signs
- **Video Mode** — Upload a video file, shows detection bar chart
- **Webcam Mode** — Live real-time detection from your camera
- Adjustable confidence threshold and image size

---

## 🛠️ Requirements

- Python 3.10+
- See `requirements.txt` for all packages

---

## ⚠️ Windows Users

Use this command to run the app (avoids PATH issues):
```bash
python -m streamlit run app.py
```

---

## 📚 References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_news.html)
- [OpenCV](https://docs.opencv.org)
- [Streamlit](https://docs.streamlit.io)
```

---

**Your `requirements.txt` (already have this but confirm it looks like this):**

```
ultralytics>=8.0.0
opencv-python>=4.8.0
streamlit>=1.30.0
Pillow>=9.0.0
numpy>=1.24.0
```

---

## How to push to GitHub

Run these commands in PowerShell inside your project folder:

```powershell
git init
git add .
git commit -m "Initial commit - Traffic Sign Detection with YOLOv8"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/traffic-sign-detection.git
git push -u origin main
```

> **Note:** Create the repo on GitHub first (github.com → New Repository), then replace `YOUR_USERNAME` with your actual GitHub username.

---

**One important thing** — add a `.gitignore` file so you don't accidentally upload `best.pt` (it's too large for GitHub) or other junk:

```
# .gitignore
best.pt
*.pt
__pycache__/
*.pyc
.streamlit/
runs/
*.egg-info/
```

Then anyone who clones your repo just needs to train on Kaggle and download their own `best.pt`.The environment is still having issues, but I can give you everything you need right here. You need two files for GitHub — a `README.md` and a `requirements.txt`. Copy and save them:

---

**Save this as `README.md` in your project folder:**

```markdown
# 🚦 Traffic Sign Detection System

A real-time traffic sign detection system using **YOLOv8**, **OpenCV**, and **Streamlit**.  
Detects and classifies **43 types** of German traffic signs from images, videos, and live webcam.

---

## 📁 Project Structure

```
├── kaggle_train_gtsrb.ipynb   # Train the model on Kaggle
├── app.py                     # Streamlit desktop application
├── requirements.txt           # Python dependencies
└── best.pt                    # Your trained model (download from Kaggle)
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/traffic-sign-detection.git
cd traffic-sign-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your trained model
Download `best.pt` from your Kaggle training run and place it in the project folder.

### 4. Run the app
```bash
python -m streamlit run app.py
```
Open your browser at **http://localhost:8501**

---

## 🏋️ Training the Model (Kaggle)

1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook
2. Add the dataset: **gtsrb-german-traffic-sign**
3. Enable **GPU T4** under Settings → Accelerator
4. Upload and run `kaggle_train_gtsrb.ipynb`
5. Download `runs/gtsrb/weights/best.pt` from the Output panel

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Name | GTSRB — German Traffic Sign Recognition Benchmark |
| Training Images | ~51,839 |
| Classes | 43 |
| Source | [Kaggle GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) |

---

## 🎯 Model Performance

| Configuration | Expected mAP50 |
|--------------|----------------|
| YOLOv8n, 30 epochs, img=416 | ~82–87% |
| YOLOv8s, 50 epochs, img=640 | ~92–95% |

---

## 🖥️ App Features

- **Image Mode** — Upload a photo and detect signs
- **Video Mode** — Upload a video file, shows detection bar chart
- **Webcam Mode** — Live real-time detection from your camera
- Adjustable confidence threshold and image size

---

## 🛠️ Requirements

- Python 3.10+
- See `requirements.txt` for all packages

---

## ⚠️ Windows Users

Use this command to run the app (avoids PATH issues):
```bash
python -m streamlit run app.py
```

---

## 📚 References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_news.html)
- [OpenCV](https://docs.opencv.org)
- [Streamlit](https://docs.streamlit.io)
```

---

**Your `requirements.txt` (already have this but confirm it looks like this):**

```
ultralytics>=8.0.0
opencv-python>=4.8.0
streamlit>=1.30.0
Pillow>=9.0.0
numpy>=1.24.0
```

---

## How to push to GitHub

Run these commands in PowerShell inside your project folder:

```powershell
git init
git add .
git commit -m "Initial commit - Traffic Sign Detection with YOLOv8"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/traffic-sign-detection.git
git push -u origin main
```

> **Note:** Create the repo on GitHub first (github.com → New Repository), then replace `YOUR_USERNAME` with your actual GitHub username.

---

**One important thing** — add a `.gitignore` file so you don't accidentally upload `best.pt` (it's too large for GitHub) or other junk:

```
# .gitignore
best.pt
*.pt
__pycache__/
*.pyc
.streamlit/
runs/
*.egg-info/
```

Then anyone who clones your repo just needs to train on Kaggle and download their own `best.pt`.
