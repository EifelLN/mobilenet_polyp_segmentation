import os
import torch
import numpy as np
import cv2
import gradio as gr
from PIL import Image
import albumentations as A

from src.models.student import StudentModel

# ===== Config =====
MODEL_PATH = "checkpoints/Best_kd.pth" 
IMG_SIZE = 320
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load Model =====
def load_model():
    """Load model yang sudah di-train."""
    model = StudentModel(encoder_name="mobilenet_v2")
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Model loaded from: {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
    
    model.to(DEVICE)
    model.eval()
    return model

# Load model saat startup
print(f"Loading model on {DEVICE}...")
model = load_model()

# ===== Preprocessing =====
def preprocess_image(image):
    """Preprocess gambar input untuk model."""
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize image
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
    ])
    transformed = transform(image=image)
    image_resized = transformed['image']
    
    # Convert to tensor (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image_resized

# ===== Inference =====
def predict(image):
    """
    Melakukan prediksi segmentasi polyp pada gambar input.
    
    Args:
        image: Gambar input (PIL Image atau numpy array)
    
    Returns:
        Tuple dari (original image, segmentation mask, overlay image)
    """
    if image is None:
        return None, None, None
    
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
    
    return image_resized, mask_colored, overlay

def create_overlay(image, mask):
    """Create overlay visualization dengan mask di atas gambar original."""
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

# ===== Gradio Interface =====
# Title dan Description
TITLE = "Polyp Segmentation dengan Knowledge Distillation"

DESCRIPTION = """

Project ini mengembangkan model **deep learning** untuk **segmentasi polyp** pada gambar kolonoskopi. 
Polyp adalah pertumbuhan abnormal pada dinding usus besar yang dapat berkembang menjadi kanker kolorektal.

### Apa yang Bisa Dilakukan Model Ini?
- **Mendeteksi** keberadaan polyp pada gambar kolonoskopi
- **Mensegmentasi** area polyp dengan akurasi tinggi
- **Real-time inference** untuk aplikasi praktis

### Teknologi yang Digunakan
- **Arsitektur**: U-Net dengan MobileNetV2 backbone (lightweight & efficient)
- **Training Method**: Knowledge Distillation dari teacher model (ResNet50)
- **Framework**: PyTorch + Segmentation Models PyTorch (SMP)

### Performa Model
Model ini dilatih menggunakan teknik **Knowledge Distillation**, dimana:
- **Teacher Model**: ResNet50 (model besar, akurasi tinggi)
- **Student Model**: MobileNetV2 (model ringan, inference cepat)

Student model belajar dari teacher model untuk mencapai akurasi yang mendekati teacher 
dengan ukuran model yang jauh lebih kecil.

---

## Cara Menggunakan

1. **Upload Gambar**: Klik area upload atau drag & drop gambar kolonoskopi
2. **Tunggu Proses**: Model akan memproses gambar secara otomatis
3. **Lihat Hasil**: 
   - **Gambar Original**: Gambar input yang telah di-resize
   - **Segmentation Mask**: Area polyp yang terdeteksi (warna panas = polyp)
   - **Overlay**: Visualisasi polyp pada gambar original (merah = polyp, hijau = kontur)

### Tips
- Gunakan gambar kolonoskopi dengan kualitas yang baik
- Model dioptimalkan untuk ukuran 320x320 pixel
- Hasil terbaik pada gambar dengan pencahayaan yang cukup

---
"""

ARTICLE = """
### Referensi Dataset
Model ini dilatih menggunakan dataset polyp segmentation publik:
- **Kvasir-SEG**
- **CVC-ClinicDB**
- **CVC-ColonDB**
- **CVC-300**
- **ETIS-LaribPolypDB**

"""

# Example images (jika ada)
EXAMPLES = []
example_paths = [
    "dataset/TestDataset/Kvasir/images",
    "dataset/TestDataset/CVC-ClinicDB/images",
]

for example_path in example_paths:
    if os.path.exists(example_path):
        images = os.listdir(example_path)[:2]  # Ambil 2 contoh dari tiap dataset
        for img in images:
            EXAMPLES.append(os.path.join(example_path, img))

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Polyp Segmentation Demo") as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Gambar Kolonoskopi",
                type="numpy",
                height=475
            )
            submit_btn = gr.Button("Analisis Gambar", variant="primary", size="lg")
            clear_btn = gr.Button("Clear", size="sm")
        
        with gr.Column(scale=2):
            with gr.Row():
                output_original = gr.Image(
                    label="Gambar Original (Resized)",
                    height=250
                )
                output_mask = gr.Image(
                    label="Segmentation Mask",
                    height=250
                )
            output_overlay = gr.Image(
                label="Overlay (Polyp Detection)",
                height=300
            )
    
    # Examples section
    if EXAMPLES:
        gr.Markdown("### Contoh Gambar")
        gr.Examples(
            examples=EXAMPLES,
            inputs=input_image,
            outputs=[output_original, output_mask, output_overlay],
            fn=predict,
            cache_examples=False
        )
    
    gr.Markdown(ARTICLE)
    
    # Event handlers
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_original, output_mask, output_overlay]
    )
    
    input_image.change(
        fn=predict,
        inputs=input_image,
        outputs=[output_original, output_mask, output_overlay]
    )
    
    clear_btn.click(
        fn=lambda: (None, None, None, None),
        outputs=[input_image, output_original, output_mask, output_overlay]
    )

# Launch app
if __name__ == "__main__":
    print("Starting Gradio App...")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set True untuk mendapatkan public URL
        show_error=True
    )
