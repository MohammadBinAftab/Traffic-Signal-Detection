# 🚦 Traffic Sign Detection System with Adaptive Q-Learning Threshold

An interactive web application that performs real-time traffic sign detection using **YOLOv8** (trained on the GTSRB dataset) and adaptively controls the confidence threshold using a **Q-Learning Reinforcement Learning agent**.

Built with **Streamlit**, **OpenCV**, and **Ultralytics**.

---

## 🌟 Features

*   **Multi-Scale Inference (Robust Mode)**: Automatically runs detection across multiple image sizes (`160`, `224`, `416`, `640`) and merges overlapping detections using **Non-Maximum Suppression (NMS)**. This ensures robust detection of traffic signs at various distances, crops, and resolutions.
*   **Adaptive RL Agent**: Integrates a tabular Q-Learning agent that dynamically adapts the YOLO confidence threshold based on frame brightness, blur levels, and recent detection counts to minimize false positives and negatives.
*   **Three Input Channels**:
    *   📷 **Image Upload**: Upload and inspect static images.
    *   🎬 **Video Processing**: Process video files frame-by-frame.
    *   📹 **Webcam**: Run real-time detection via your local webcam.
*   **Automatic Model Loading**: Automatically searches for and loads `best.pt` from the project directory.

---

## 📁 Project Structure

```
├── app.py                     # Streamlit desktop application
├── rl_agent.py                # Adaptive Reinforcement Learning agent (Q-Learning)
├── q_table.json               # Saved state-action values for RL agent
├── requirements.txt           # Python dependencies
├── best.pt                    # Trained YOLOv8 model weights
└── kaggle_train_gtsrb.ipynb   # Jupyter Notebook to train the model on Kaggle
```

---

## 🛠️ Setup Instructions (For All Laptops)

Follow the setup steps corresponding to your operating system.

### Prerequisites

Ensure you have **Python 3.8 to 3.11** installed on your laptop. (Python 3.12+ is also supported, but 3.8-3.11 is recommended for torch/webcam package compatibility).
- **Windows**: Download the installer from [python.org](https://www.python.org/downloads/) (make sure to check "Add Python to PATH").
- **macOS**: Install via Homebrew: `brew install python`
- **Linux (Ubuntu/Debian)**: Run `sudo apt update && sudo apt install python3 python3-pip python3-venv`

---

### Step 1: Clone the Repository

Open your terminal or command prompt and clone the repository:

```bash
git clone https://github.com/MohammadBinAftab/Traffic-Signal-Detection.git
cd Traffic-Signal-Detection
```

---

### Step 2: Set Up a Virtual Environment (Recommended)

Creating a virtual environment ensures that the project dependencies do not conflict with other Python packages on your laptop.

*   **Windows (PowerShell)**:
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
*   **Windows (Command Prompt)**:
    ```cmd
    python -m venv venv
    .\venv\Scripts\activate.bat
    ```
*   **macOS / Linux**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

---

### Step 3: Install Dependencies

With the virtual environment activated, run the following command to install all required libraries:

```bash
pip install -r requirements.txt
```

---

### Step 4: Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The app will start a local server and output the access URLs. Open your browser and navigate to:
👉 **`http://localhost:8501`**

---

## 🏋️ Training the Model (Kaggle)

If you wish to re-train the YOLOv8 model on the GTSRB dataset:
1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook.
2. Add the dataset: **gtsrb-german-traffic-sign**.
3. Enable **GPU T4** under Settings → Accelerator.
4. Upload and run `kaggle_train_gtsrb.ipynb`.
5. Download `runs/gtsrb/weights/best.pt` from the Output panel and place it in the project root directory.

---

## 📊 Dataset & Model Performance

### Dataset Details

| Property | Value |
|----------|-------|
| Name | GTSRB — German Traffic Sign Recognition Benchmark |
| Training Images | ~51,839 |
| Classes | 43 |
| Source | [Kaggle GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) |

### Expected Accuracy

| Configuration | Expected mAP50 |
|--------------|----------------|
| YOLOv8n, 30 epochs, img=416 | ~82–87% |
| YOLOv8s, 50 epochs, img=640 | ~92–95% |

---

## 🔧 Troubleshooting

### 1. Camera Access Error on Webcam Tab
*   Make sure no other program (Zoom, Teams, Skype, etc.) is using your webcam.
*   If you have external webcams, you may need to edit line 286 in `app.py` (`cv2.VideoCapture(0)`) to change the camera index (e.g. `cv2.VideoCapture(1)`).

### 2. Missing C++ Build Tools (Windows)
If installing dependencies throws an error compiling certain libraries, you may need to install the Visual C++ Redistributable. Download it from the [Official Microsoft Support Page](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist).

### 3. PyTorch CUDA / GPU Support
By default, `pip install -r requirements.txt` installs the CPU version of PyTorch. If you have an Nvidia GPU and want faster real-time webcam inference:
1. Uninstall torch: `pip uninstall torch torchvision`
2. Install the CUDA-enabled version of PyTorch by following instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

---

## 📚 References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_news.html)
- [OpenCV](https://docs.opencv.org)
- [Streamlit](https://docs.streamlit.io)
