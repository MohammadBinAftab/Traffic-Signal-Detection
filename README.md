# 🚦 Traffic Sign Detection with Adaptive Q-Learning Threshold

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

## 🛠️ Setup Instructions (For All Laptops)

Follow the setup steps corresponding to your operating system.

### Prerequisites

Ensure you have **Python 3.8 to 3.11** installed on your laptop. (Python 3.12+ is also supported, but 3.8-3.11 is recommended for torch/webcam package compatibility).
- **Windows**: Download the installer from [python.org](https://www.python.org/downloads/) (make sure to check "Add Python to PATH").
- **macOS**: Install via Homebrew: `brew install python`
- **Linux (Ubuntu/Debian)**: Run `sudo apt update && sudo apt install python3 python3-pip python3-venv`

---

### Step 1: Clone or Download the Repository

Open your terminal or command prompt and clone the repository:

```bash
git clone https://github.com/MohammadBinAftab/Traffic-Signal-Detection.git
cd Traffic-Signal-Detection
```

If downloading as a ZIP file:
1. Extract the folder.
2. Open terminal/powershell and navigate to the directory (e.g., `cd path/to/Traffic-Signal-Detection`).

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

*Note: This will install `streamlit`, `ultralytics` (YOLOv8), `opencv-python`, `Pillow`, and `numpy`.*

---

### Step 4: Add the Model Weights

Make sure you have your trained model weights file named **`best.pt`** placed directly in the project root directory alongside `app.py`. The app will auto-detect and load it on launch.

---

### Step 5: Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The app will start a local server and output the access URLs. By default, it will open in your web browser at:
👉 **`http://localhost:8501`**

---

## 💡 Using the App

1.  **Select Inference Mode**:
    *   **`Multi-Scale (Robust)`** (Default): Recommended for most images. It evaluates the frame at multiple resolutions to capture signs regardless of how small or large they are.
    *   **Single Resolutions (`160` to `640`)**: Runs inference at a single fixed size. Lower sizes (like `160` or `224`) are fast and work best for cropped/clean graphics. Higher sizes (like `640`) are better for distant signs in high-res wide shots.
2.  **Adaptive Threshold vs. Manual Slider**:
    *   Toggle **Use RL Adaptive Threshold** in the sidebar to let the Q-learning agent choose the confidence threshold.
    *   Turn it off to manually specify the confidence threshold using the slider.
3.  **Real-Time Webcam**:
    *   Navigate to the **Webcam** tab, toggle "Start Webcam", and allow camera access. Click "Stop Webcam" to release the camera.

---

## 📂 Project Structure

*   `app.py`: Main Streamlit web application interface, image pre-processing, and multi-scale inference engine.
*   `rl_agent.py`: Adaptive Reinforcement Learning agent (Q-Learning) implementation.
*   `best.pt`: Pre-trained YOLOv8 weights (GTSRB dataset).
*   `q_table.json`: JSON file storing the learned state-action Q-values for the RL agent.
*   `requirements.txt`: Python package dependencies list.

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
