import streamlit as st
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import time
import pandas as pd

# ==========================================
# SYSTEM CONFIGURATION
# ==========================================
YOLO_MODEL_PATH = 'best2511.pt' 
RESNET_MODEL_PATH = 'pin_classifier_resnet50_best2511.pt'

# Fixed Threshold for ResNet (Accept only if confidence > 0.8)
RESNET_CONF_THRESHOLD = 0.8 

# 1. Define classes for ResNet
RESNET_CLASSES = {
    0: "Bad",
    1: "Good",
    2: "Unknown"
}

# Helper function: Get color based on status and threshold
def get_status_color(status_label, confidence, threshold):
    # If AI is not confident (confidence < threshold) -> Mark as Unknown
    if confidence < threshold:
        return "Unknown", (0, 255, 255) # Yellow

    # If confident, return the predicted label color
    if status_label == "Good":
        return "Good", (0, 255, 0)      # Green
    elif status_label == "Bad":
        return "Bad", (0, 0, 255)       # Red
    else:
        return "Unknown", (0, 255, 255) # Yellow

# ==========================================
# UI SETUP & MODEL LOADING
# ==========================================

st.set_page_config(page_title="Quality Inspection App", layout="wide")
st.title("🏭 Quality Inspection: Terminal Blocks & Resistors")
st.markdown("---")

# Load Models with Caching
@st.cache_resource
def load_yolo_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Error loading YOLO: {e}")
        return None

@st.cache_resource
def load_resnet_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = models.resnet50()
        
        # --- FIX KEY MISMATCH ERROR ---
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(num_ftrs, len(RESNET_CLASSES)) 
        )
        # ---------------------------------

        checkpoint = torch.load(path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model.to(device)
    except Exception as e:
        st.error(f"Error loading ResNet: {e}")
        return None

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize Models
yolo_model = load_yolo_model(YOLO_MODEL_PATH)
resnet_model = load_resnet_model(RESNET_MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# SIDEBAR SETTINGS
# ==========================================
st.sidebar.header("⚙️ Configuration")
conf_threshold = st.sidebar.slider("YOLO Detection Sensitivity", 0.0, 1.0, 0.4, 0.05)
use_webcam = st.sidebar.checkbox("Start Webcam", value=True)

if yolo_model:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📦 Detectable Objects:")
    st.sidebar.json(yolo_model.names)

# ==========================================
# MAIN LAYOUT (Video + Dashboard)
# ==========================================
# Create two columns: Left for Video (70%), Right for Stats (30%)
col_video, col_stats = st.columns([0.7, 0.3])

with col_video:
    st.subheader("📹 Live Camera Feed")
    image_placeholder = st.empty()

with col_stats:
    st.subheader("📊 Live Statistics")
    # This container will hold our dynamic columns
    stats_container = st.empty()

# ==========================================
# INFERENCE LOGIC
# ==========================================

def run_inference():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): 
        st.error("Cannot access Webcam!")
        return

    # Get list of class names from YOLO
    yolo_names = yolo_model.names 

    while cap.isOpened() and use_webcam:
        ret, frame = cap.read()
        if not ret: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- INITIALIZE STATS FOR CURRENT FRAME ---
        # Create a dictionary to hold stats for EACH class separately
        # Structure: { "Terminal": {"Good":0, "Bad":0, ...}, "Resistor": {...} }
        frame_stats = {name: {"Good": 0, "Bad": 0, "Unknown": 0} for name in yolo_names.values()}

        # 1. Run YOLO object detection
        results = yolo_model(img_rgb, conf=conf_threshold, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get Object Name from YOLO
                cls_id = int(box.cls[0])
                object_name = yolo_names[cls_id]

                # Crop image for ResNet
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1 and resnet_model:
                    crop_img = img_rgb[y1:y2, x1:x2]
                    pil_img = Image.fromarray(crop_img)
                    
                    # 2. Run ResNet for Status Classification
                    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = resnet_model(input_tensor)
                        probs = torch.nn.functional.softmax(output[0], dim=0)
                        
                        top_p, top_class = probs.topk(1)
                        resnet_conf = float(top_p.cpu().numpy()[0])
                        resnet_idx = int(top_class.cpu().numpy()[0])

                    # Get Raw Status
                    raw_status = RESNET_CLASSES.get(resnet_idx, "Unknown")
                    
                    # 3. Determine Final Status (Apply Threshold)
                    final_status, color = get_status_color(raw_status, resnet_conf, RESNET_CONF_THRESHOLD)

                    # --- UPDATE STATS (Specific to Object Type) ---
                    if object_name in frame_stats and final_status in frame_stats[object_name]:
                        frame_stats[object_name][final_status] += 1

                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw Label
                    label_text = f"{object_name} | {final_status} | {resnet_conf:.0%}"
                    font_scale = 0.6
                    thickness = 2 
                    text_color = (0, 0, 0) 

                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
                    cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        # --- UPDATE UI ELEMENTS ---
        # 1. Update Video Feed
        image_placeholder.image(frame, channels="BGR")
        
        # 2. Update Detailed Statistics
        # We use the container to refresh the columns every frame
        with stats_container.container():
            # Create dynamic columns based on number of YOLO classes
            # If you have 5 classes, this creates 5 columns
            cols = st.columns(len(yolo_names))
            
            for idx, (cls_id, cls_name) in enumerate(yolo_names.items()):
                # Get stats for this specific class
                stats = frame_stats[cls_name]
                
                with cols[idx]:
                    # Display Header (Object Name)
                    st.markdown(f"**{cls_name}**")
                    st.markdown("---")
                    
                    # Display Counts
                    st.markdown(f"✅ : **{stats['Good']}**")
                    st.markdown(f"❌ : **{stats['Bad']}**")
                    st.markdown(f"⚠️ : **{stats['Unknown']}**")

        time.sleep(0.01)

    cap.release()

if use_webcam:
    run_inference()