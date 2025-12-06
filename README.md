INSTRUCTION for using my project: Terminal Block Connector Status Recognition

This repository contains a Deep Learning application designed to automatically detect and classify the status of terminal block connectors.

The system utilizes a hybrid architecture combining **YOLOv8n** for precise object detection and **ResNet50** for robust classification, achieving high accuracy in standard operational environments.

## I. System Architecture:

The workflow consists of two main stages:
1.  **Object Detection (YOLOv8n):** Locates terminal block connectors within the input image/video stream.
2.  **Classification (ResNet50):** Analyzes the detected regions to determine their specific status (e.g., Open, Closed, Loose, Secured).

## II. Project Structure:

* **`app.py`**: The main application script (User Interface).
* **`requirements.txt`**: List of all Python dependencies required (PyTorch, Ultralytics, etc.).
* **`best*.pt`** (Model Weights):
    * `best1111.pt`: Model checkpoint (Version: Nov 11, 2025).
    * `best2511.pt`: Model checkpoint (Version: Nov 25, 2025).
    * `best25112.pt`: A better version of `best2511.pt`.

## III. Prerequisites:

Ensure your system meets the following requirements:
* **Python 3.10 or 3.11**
* **Git**
* **CUDA (Optional):** Recommended for faster inference if you have an NVIDIA GPU.

## IV. Installation Guide:

### 1. Clone the Repository
Open your terminal/command prompt and run:
git clone https://github.com/duyanhtoanvui/Terminal-Block-Connector
cd "App Demo"

### 2. Create a Virtual Environment (Recommended)
Isolate your dependencies to avoid conflicts.

**Windows:**
python -m venv venv
.\venv\Scripts\activate

**macOS / Linux:**
python3 -m venv venv
source venv/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt
#### Usage
To launch the recognition system, run the command corresponding to your framework:

**If using Streamlit (Recommended for UI Demos):**
streamlit run app.py

**If using standard Python/Flask:**
python app.py

Open your browser at the provided local URL (usually http://localhost:8501) to interact with the application.

## V. Troubleshooting:
**Issue: "Untracked working tree files would be overwritten by merge"**
If you see this error when running git pull, it means your local files (like app.py or .pt weights) conflict with the remote version.

**Option A: Keep your local changes (Safe) Rename the conflicting files before pulling:**
Rename app.py -> app_backup.py.
Run git pull origin main.

**Option B: Force update (Data Loss Warning) This will discard your local changes and match the server exactly:**
git fetch --all
git reset --hard origin/main

**Issue: "ModuleNotFoundError"**
If the app crashes due to missing libraries, ensure your virtual environment is active and run:
pip install -r requirements.txt
## VI. Author: Nguyen Tan Duy Anh - HCMUS
