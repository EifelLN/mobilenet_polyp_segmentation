"""
Polyp Segmentation Web Application
===================================
This script provides a Streamlit web interface for polyp segmentation.
It supports comparing results from two models:
- Knowledge Distillation model (Best_kd.pth)
- Baseline model (Best_Baseline.pth)

Usage:
    streamlit run app.py
    python main.py app
"""

import os
import torch
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import albumentations as A

from src.models.student import StudentModel

# ===== Config =====
KD_MODEL_PATH = "checkpoints/Best_kd.pth"
BASELINE_MODEL_PATH = "checkpoints/Best_Baseline.pth"
IMG_SIZE = 320
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Load Model =====
@st.cache_resource
def load_model(model_path):
    """Load a trained model from checkpoint."""
    model = StudentModel(encoder_name="mobilenet_v2")
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Model loaded from: {model_path}")
    else:
        return None
    
    model.to(DEVICE)
    model.eval()
    return model


def get_available_models():
    """Get list of available model checkpoints."""
    models = {}
    if os.path.exists(KD_MODEL_PATH):
        models["Knowledge Distillation (Best_kd.pth)"] = KD_MODEL_PATH
    if os.path.exists(BASELINE_MODEL_PATH):
        models["Baseline (Best_Baseline.pth)"] = BASELINE_MODEL_PATH
    return models


# ===== Preprocessing =====
def preprocess_image(image):
    """Preprocess input image for the model."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure RGB format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize image
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
    ])
    transformed = transform(image=image)
    image_resized = transformed['image']
    
    # Convert to tensor (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image_resized


# ===== Inference =====
def predict(model, image):
    """
    Perform polyp segmentation prediction on input image.
    
    Args:
        model: The loaded model
        image: Input image (PIL Image or numpy array)
    
    Returns:
        Tuple of (original image, segmentation mask, overlay image, probability map)
    """
    if image is None or model is None:
        return None, None, None, None
    
    # Preprocess
    image_tensor, image_resized = preprocess_image(image)
    image_tensor = image_tensor.to(DEVICE)
    
    # Inference
    with torch.no_grad():
        logits = model(image_tensor)
        pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred_mask = (pred_prob > 0.5).astype(np.uint8) * 255
    
    # Create overlay visualization
    overlay = create_overlay(image_resized, pred_mask)
    
    # Convert mask to RGB for display
    mask_colored = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
    
    return image_resized, mask_colored, overlay, pred_prob


def create_overlay(image, mask):
    """Create overlay visualization with mask on top of original image."""
    # Create colored mask (red for polyp)
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 0] = mask  # Red channel
    
    # Blend with original image
    alpha = 0.5
    overlay = cv2.addWeighted(image, 1, mask_colored, alpha, 0)
    
    # Add contour for better visibility
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)  # Green contour
    
    return overlay


def calculate_polyp_area(pred_prob, threshold=0.5):
    """Calculate the percentage of image area detected as polyp."""
    binary_mask = (pred_prob > threshold).astype(np.float32)
    polyp_pixels = np.sum(binary_mask)
    total_pixels = pred_prob.size
    return (polyp_pixels / total_pixels) * 100


def get_example_images():
    """Get example images from test datasets."""
    examples = []
    example_paths = [
        "dataset/TestDataset/Kvasir/images",
        "dataset/TestDataset/CVC-ClinicDB/images",
    ]
    
    for example_path in example_paths:
        if os.path.exists(example_path):
            images = sorted(os.listdir(example_path))[:3]
            for img in images:
                examples.append(os.path.join(example_path, img))
    
    return examples


# ===== Streamlit App =====
def main():
    # Page configuration
    st.set_page_config(
        page_title="Polyp Segmentation",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Modern dark-themed CSS
    st.markdown("""
        <style>
        /* Hide streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Main container styling */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Header styling */
        .main-title {
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }
        
        .subtitle {
            font-size: 1.1rem;
            color: #888;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* Card styling */
        .result-card {
            background: linear-gradient(145deg, #1e1e2e 0%, #2d2d3f 100%);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        
        .model-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Image container styling */
        .image-container {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid rgba(255,255,255,0.08);
            text-align: center;
        }
        
        .image-label {
            font-size: 0.85rem;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 0.75rem;
            font-weight: 500;
        }
        
        /* Image styling */
        .stImage > img {
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        }
        
        /* Uploader styling */
        .upload-section {
            background: linear-gradient(145deg, #1a1a2e 0%, #252538 100%);
            border: 2px dashed rgba(102, 126, 234, 0.4);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: rgba(102, 126, 234, 0.8);
            background: linear-gradient(145deg, #1e1e32 0%, #2a2a40 100%);
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16161f 100%);
        }
        
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }
        
        .sidebar-header {
            font-size: 1.3rem;
            font-weight: 600;
            color: #fff;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(102, 126, 234, 0.5);
        }
        
        .info-box {
            background: rgba(102, 126, 234, 0.1);
            border-left: 3px solid #667eea;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
        }
        
        .info-box p {
            margin: 0;
            color: #ccc;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        /* Divider styling */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            margin: 1.5rem 0;
        }
        
        /* Comparison section */
        .comparison-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }
        
        .comparison-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #fff;
        }
        
        /* Status indicator */
        .status-dot {
            width: 8px;
            height: 8px;
            background: #4ade80;
            border-radius: 50%;
            display: inline-block;
            margin-right: 0.5rem;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: #666;
        }
        
        .empty-state-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }
        
        .empty-state-text {
            font-size: 1.1rem;
            color: #888;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Settings</div>', unsafe_allow_html=True)
        
        # Model selection
        available_models = get_available_models()
        
        if not available_models:
            st.error("No models found in checkpoints folder.")
            st.stop()
        
        compare_mode = st.toggle("Compare Models", value=len(available_models) > 1)
        
        if compare_mode and len(available_models) >= 2:
            selected_models = list(available_models.keys())
            st.info(f"Comparing {len(selected_models)} models")
        else:
            selected_model = st.selectbox(
                "Select Model",
                options=list(available_models.keys()),
                index=0
            )
            selected_models = [selected_model]
        
        st.markdown("---")
        
        # Device info
        st.markdown(f"""
        <div class="info-box">
            <p><strong>Device:</strong> {DEVICE}</p>
            <p><strong>Input Size:</strong> {IMG_SIZE}×{IMG_SIZE}px</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # About section
        st.markdown('<div class="sidebar-header">About</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <p>Polyp segmentation using <strong>MobileNetV2</strong> with knowledge distillation from a ResNet50 teacher model.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("Built with Streamlit • PyTorch")
    
    # Main content
    st.markdown('<h1 class="main-title">Polyp Segmentation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered medical image analysis for polyp detection</p>', unsafe_allow_html=True)
    
    # Upload section
    st.markdown("### Upload Image")
    
    col_upload, col_examples = st.columns([3, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=["png", "jpg", "jpeg", "bmp"],
            help="Supported formats: PNG, JPG, JPEG, BMP"
        )
    
    with col_examples:
        example_images = get_example_images()
        if example_images:
            example_names = [os.path.basename(p) for p in example_images]
            selected_example = st.selectbox(
                "Examples",
                options=["Select..."] + example_names,
                help="Choose from test dataset"
            )
        else:
            selected_example = "Select..."
    
    # Determine which image to use
    input_image = None
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
    elif selected_example != "Select...":
        idx = example_names.index(selected_example)
        input_image = Image.open(example_images[idx])
    
    st.markdown("---")
    
    # Process and display results
    if input_image is not None:
        
        if compare_mode and len(available_models) >= 2:
            # Comparison mode
            st.markdown("### Model Comparison")
            
            # Load models and get predictions
            model_names = list(available_models.keys())
            model_paths = list(available_models.values())
            
            results = []
            with st.spinner("Processing with multiple models..."):
                for name, path in zip(model_names, model_paths):
                    model = load_model(path)
                    if model is not None:
                        original, mask, overlay, prob = predict(model, input_image)
                        area = calculate_polyp_area(prob) if prob is not None else 0
                        results.append({
                            "name": name,
                            "original": original,
                            "mask": mask,
                            "overlay": overlay,
                            "prob": prob,
                            "area": area
                        })
            
            # Show original image first
            st.markdown("#### Original Image")
            col_orig, col_space = st.columns([1, 2])
            with col_orig:
                st.image(results[0]["original"], use_container_width=True)
            
            st.markdown("---")
            
            # Display comparison in columns
            cols = st.columns(len(results))
            
            for i, (col, result) in enumerate(zip(cols, results)):
                with col:
                    model_short_name = result["name"].split(" (")[0]
                    
                    # Model header with metric
                    st.markdown(f"""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <span class="model-badge">{model_short_name}</span>
                        <div style="margin-top: 0.75rem;">
                            <span class="metric-value">{result['area']:.1f}%</span>
                            <br>
                            <span class="metric-label">Polyp Area</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mask
                    st.image(result["mask"], use_container_width=True)
                    st.markdown('<p class="image-label">Segmentation Mask</p>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Overlay
                    st.image(result["overlay"], use_container_width=True)
                    st.markdown('<p class="image-label">Overlay</p>', unsafe_allow_html=True)
        
        else:
            # Single model mode
            st.markdown("### Segmentation Results")
            
            # Load selected model
            model_name = selected_models[0]
            model_path = available_models[model_name]
            model = load_model(model_path)
            
            if model is None:
                st.error(f"Failed to load model: {model_path}")
                st.stop()
            
            # Run prediction
            with st.spinner("Analyzing image..."):
                original, mask, overlay, prob = predict(model, input_image)
            
            if original is not None:
                area = calculate_polyp_area(prob)
                
                # Model info and metric
                col_info, col_metric = st.columns([2, 1])
                with col_info:
                    model_short_name = model_name.split(" (")[0]
                    st.markdown(f'<span class="model-badge">{model_short_name}</span>', unsafe_allow_html=True)
                with col_metric:
                    st.metric("Polyp Area", f"{area:.1f}%")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Display images in nice grid
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(original, use_container_width=True)
                    st.markdown('<p class="image-label">Original</p>', unsafe_allow_html=True)
                
                with col2:
                    st.image(mask, use_container_width=True)
                    st.markdown('<p class="image-label">Segmentation Mask</p>', unsafe_allow_html=True)
                
                with col3:
                    st.image(overlay, use_container_width=True)
                    st.markdown('<p class="image-label">Overlay</p>', unsafe_allow_html=True)
            else:
                st.error("Failed to process image.")
    
    else:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <p class="empty-state-text">Upload an image or select an example to get started</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
