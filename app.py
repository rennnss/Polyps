import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set page config with a more appealing theme
st.set_page_config(
    page_title="Polyps Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: #121212;
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Text */
    p, div, span {
        color: #ffffff;
        font-size: 1rem;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #2196F3;
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 16px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #1976D2;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* File uploader */
    .stFileUploader>div>div>div>div {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 24px;
        border: 2px dashed #424242;
        transition: all 0.3s ease;
    }
    .stFileUploader>div>div>div>div:hover {
        border-color: #2196F3;
    }
    
    /* Cards and containers */
    .css-1d391kg {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #1E1E1E;
        border: 1px solid #424242;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #1E1E1E;
        padding: 2rem;
    }
    
    /* Tables */
    .stDataFrame {
        background-color: #1E1E1E;
        border-radius: 8px;
        overflow: hidden;
    }
    .stDataFrame th {
        background-color: #2D2D2D;
        color: #ffffff;
        font-weight: 600;
        padding: 12px;
    }
    .stDataFrame td {
        color: #ffffff;
        padding: 12px;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #1E1E1E;
        border: 1px solid #424242;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stMetric .label {
        color: #B0B0B0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .stMetric .value {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 1rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    /* Spinner */
    .stSpinner>div {
        color: #2196F3;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1E1E1E;
        border: 1px solid #424242;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #ffffff;
    }
    
    /* Selectbox and other inputs */
    .stSelectbox, .stTextInput, .stNumberInput {
        background-color: #1E1E1E;
        color: #ffffff;
    }
    
    /* Divider */
    .stMarkdown hr {
        border-color: #424242;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🔍 About")
    st.markdown("""
    This application uses ultraltics library for real-time/image polyps detection.
    Upload an image and click 'Detect polyps' to see the results.
    """)
    
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload an image using the file uploader
    2. Click the 'Detect polyps' button
    3. View the detection results and statistics
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ Model Information")
    st.markdown("""
    - **Model**: YOLO v8
    - **Version**: 8.1.28
    - **Framework**: PyTorch
    - **Task**: Detection of polyps in images
    """)

# Main content
st.title("Polyps Detection")
st.markdown("### 📸 Upload an image to detect polyps")

# Load the model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# File uploader with better styling
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Create two columns for the images
    col1, col2 = st.columns(2)
    
    # Read the image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    with col1:
        st.markdown("### Original Image")
        st.image(image, use_column_width=True)
    
    # Run inference
    if st.button("Detect Polyps", key="detect_button"):
        with st.spinner("🔍 Detecting Polyps... This may take a few seconds."):
            # Run YOLO inference
            results = model(image_np)
            
            # Get the annotated image
            annotated_image = results[0].plot()
            
            # Convert BGR to RGB
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown("### Detection Results")
                st.image(annotated_image, use_column_width=True)
            
            # Display detection results in an expandable section
            with st.expander("📊 Detailed Detection Results", expanded=True):
                st.markdown("### Detected Polyps")
                
                # Create a table for the results
                detection_data = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[class_id]
                        detection_data.append({
                            "Object": class_name,
                            "Confidence": f"{confidence:.2%}",
                            "Box Coordinates": f"{box.xyxy[0].tolist()}"
                        })
                
                if detection_data:
                    st.table(detection_data)
                else:
                    st.info("No Polyps detected in the image.")
                
                # Add some statistics
                st.markdown("### 📈 Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Objects Detected", len(detection_data))
                with col2:
                    if detection_data:
                        avg_confidence = sum(float(d["Confidence"].strip('%'))/100 for d in detection_data) / len(detection_data)
                        st.metric("Average Confidence", f"{avg_confidence:.2%}") 