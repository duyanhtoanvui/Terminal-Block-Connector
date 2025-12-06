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

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
YOLO_MODEL_PATH = 'best2511.pt'  
RESNET_MODEL_PATH = 'pin_classifier_resnet50_best2511.pt'

# 1. Định nghĩa lại nhãn cho ResNet (Theo đúng nghiệp vụ mới)
# Model ResNet bây giờ chỉ trả lời câu hỏi: Tình trạng là gì?
# QUAN TRỌNG: Bạn cần kiểm tra lại xem lúc train ResNet bạn đặt folder nào là 0, 1, 2?
# Thường xếp theo A-Z: 0: Bad, 1: Good, 2: Unknown (Ví dụ vậy)
RESNET_CLASSES = {
    0: "Bad",
    1: "Good",
    2: "Unknown"
}

# Hàm xác định màu sắc dựa trên kết quả ResNet
def get_status_color(status_label, confidence, threshold):
    # Nếu AI không chắc chắn (độ tin cậy thấp hơn ngưỡng user cài đặt)
    # Thì dù AI đoán là Good hay Bad, ta vẫn coi là Unknown (Cần người kiểm tra)
    if confidence < threshold:
        return "Unknown", (0, 255, 255) # Màu Vàng

    # Nếu độ tin cậy cao, lấy đúng nhãn AI phán đoán
    if status_label == "Good":
        return "Good", (0, 255, 0)      # Màu Xanh lá
    elif status_label == "Bad":
        return "Bad", (0, 0, 255)       # Màu Đỏ
    else:
        return "Unknown", (0, 255, 255) # Màu Vàng (Cho class Unknown gốc)

# ==========================================
# SETUP GIAO DIỆN & MODEL
# ==========================================

st.set_page_config(page_title="App Demo - QC Check", layout="wide")
st.title("App Demo: Nhận diện Terminal Block & Resistor")
st.markdown("---")

c1, c2 = st.columns(2)
c1.info(f"YOLO (Tìm vật thể): {YOLO_MODEL_PATH}")
c2.info(f"ResNet (Check lỗi): {RESNET_MODEL_PATH}")

@st.cache_resource
def load_yolo_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Lỗi load YOLO: {e}")
        return None

@st.cache_resource
def load_resnet_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = models.resnet50()
        
        # --- PHẦN SỬA LỖI KEY MISMATCH ---
        # Lỗi "fc.1.weight" nghĩa là lớp fc là một chuỗi Sequential(Dropout, Linear)
        # Chứ không phải chỉ là 1 lớp Linear đơn lẻ.
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2), # Lớp 0 (Không có trọng số nên không báo lỗi thiếu key)
            nn.Linear(num_ftrs, len(RESNET_CLASSES)) # Lớp 1 (Chính là fc.1.weight)
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
        st.error(f"Lỗi load ResNet: {e}. (LƯU Ý: Nếu lỗi size mismatch, hãy kiểm tra lại ResNet của bạn train mấy class?)")
        return None

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==========================================
# MAIN APP
# ==========================================

yolo_model = load_yolo_model(YOLO_MODEL_PATH)
resnet_model = load_resnet_model(RESNET_MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- SIDEBAR DEBUG ---
st.sidebar.header("Cấu hình tham số")
conf_threshold = st.sidebar.slider("YOLO Threshold (Độ nhạy phát hiện)", 0.0, 1.0, 0.4, 0.05)
cls_threshold = st.sidebar.slider("ResNet Threshold (Độ chắc chắn)", 0.0, 1.0, 0.7, 0.05)
use_webcam = st.sidebar.checkbox("Sử dụng Webcam", value=True)

if yolo_model:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 YOLO đã học các class:")
    # Hiển thị danh sách class mà YOLO biết để user kiểm tra
    st.sidebar.json(yolo_model.names)
# ---------------------

image_placeholder = st.empty()

def run_inference():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): 
        st.error("Không bật được Webcam")
        return

    # Lấy danh sách tên vật thể từ YOLO (Ví dụ: 0: Terminal, 1: Resistor...)
    # Cái này có sẵn trong file .pt của YOLO
    yolo_names = yolo_model.names 

    while cap.isOpened() and use_webcam:
        ret, frame = cap.read()
        if not ret: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Dùng YOLO để tìm vật thể
        results = yolo_model(img_rgb, conf=conf_threshold, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Lấy tọa độ
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Lấy tên vật thể từ YOLO (Ví dụ: "Resistor 100K")
                cls_id = int(box.cls[0])
                object_name = yolo_names[cls_id]

                # Crop ảnh để đưa vào ResNet check Good/Bad
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1 and resnet_model:
                    crop_img = img_rgb[y1:y2, x1:x2]
                    pil_img = Image.fromarray(crop_img)
                    
                    # 2. Dùng ResNet để check trạng thái
                    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = resnet_model(input_tensor)
                        probs = torch.nn.functional.softmax(output[0], dim=0)
                        
                        # Lấy class có xác suất cao nhất
                        top_p, top_class = probs.topk(1)
                        resnet_conf = float(top_p.cpu().numpy()[0])
                        resnet_idx = int(top_class.cpu().numpy()[0])

                    # Lấy nhãn trạng thái (Good/Bad/Unknown)
                    raw_status = RESNET_CLASSES.get(resnet_idx, "Unknown")
                    
                    # 3. Quyết định màu sắc và nhãn hiển thị
                    final_status, color = get_status_color(raw_status, resnet_conf, cls_threshold)

                    # Vẽ lên màn hình
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Hiển thị: Tên vật thể (YOLO) | Trạng thái (ResNet) | Độ tin cậy (ResNet)
                    label_text = f"{object_name} | {final_status} | {resnet_conf:.0%}"
                    
                    # Cấu hình font chữ
                    font_scale = 0.6
                    thickness = 2  # ĐỘ ĐẬM
                    text_color = (0, 0, 0) # MÀU ĐEN

                    # Vẽ nền cho chữ dễ đọc
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
                    
                    # Vẽ chữ
                    cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        image_placeholder.image(frame, channels="BGR")
        time.sleep(0.01)

    cap.release()

if use_webcam:
    run_inference()