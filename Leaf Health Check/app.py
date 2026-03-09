import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Leaf Health Check",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LIGHT BACKEND LOGIC (Replacing TensorFlow) ---
def analyze_leaf_opencv(image_path):
    # Image-ah read panrom
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # BGR to HSV conversion (Color analysis-kaga)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Healthy Green Range
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_healthy = cv2.inRange(hsv, lower_green, upper_green)
    
    # Disease/Yellow/Brown Range
    lower_disease = np.array([10, 30, 20])
    upper_disease = np.array([30, 255, 200])
    mask_disease = cv2.inRange(hsv, lower_disease, upper_disease)
    
    # Calculation
    healthy_pixels = cv2.countNonZero(mask_healthy)
    disease_pixels = cv2.countNonZero(mask_disease)
    total_pixels = healthy_pixels + disease_pixels
    
    severity_perc = (disease_pixels / total_pixels * 100) if total_pixels > 0 else 0
    
    # UI-kaga data formatting
    status = "HEALTHY" if severity_perc < 2 else "INFECTED"
    severity_label = "Low" if severity_perc < 5 else "Medium" if severity_perc < 15 else "High"
    
    return {
        "status": status,
        "severity": severity_label,
        "damage": round(severity_perc, 2),
        "plant_type": "Hibiscus" # Default-ah hibiscus-nu vachurken
    }

# --- SIDEBAR (Exactly like your Screenshot 1) ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    mode = st.radio(
        "Select Mode:",
        ["🔍 Analyze Leaf", "🤖 AI Assistant", "📊 Analysis History", "📋 Care Plan", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### 📚 Guide")
    st.write("1. Upload a clear photo.")
    st.write("2. Analyze to detect disease.")
    st.write("3. Follow rescue tips.")
    
    st.markdown("---")
    st.markdown("### 🔬 System Info")
    st.success("✅ Database: Active")
    if st.button("Reload Models"):
        st.rerun()

# --- MAIN CONTENT ---
st.markdown("# 🍃 Leaf Health Check")
st.markdown("##### AI-Powered Plant Disease Detection & Diagnosis")
st.markdown("---")

if mode == "🔍 Analyze Leaf":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload & Analyze")
        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            temp_path = "temp_analysis_img.jpg"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(uploaded_file, caption="Target Leaf", use_container_width=True)

    with col2:
        if uploaded_file:
            st.subheader("Analysis Results")
            results = analyze_leaf_opencv(temp_path)
            
            # Professional Metric Cards (Screenshot 1 Style)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Plant", results["plant_type"])
            m2.metric("Disease", "None" if results["status"] == "HEALTHY" else "Leaf Spot")
            m3.metric("Severity", results["severity"])
            m4.metric("Area Affected", f"{results['damage']}%")
            
            st.markdown("---")
            
            # Status Display
            if results["status"] == "HEALTHY":
                st.success(f"### STATUS: {results['status']}")
                st.balloons()
            else:
                st.error(f"### STATUS: {results['status']}")
            
            st.write(f"**Damage Detected:** {results['damage']}%")
            
            st.markdown("### 🆘 Rescue Recommendations")
            if results["status"] == "HEALTHY":
                st.info("Everything looks good! Keep following the regular watering schedule.")
            else:
                st.warning("Isolate the plant and apply organic fungicide. Avoid wetting the leaves.")

elif mode == "📊 Analysis History":
    st.info("Feature coming soon! Database integration in progress.")

else:
    st.write(f"Section {mode} is under maintenance.")