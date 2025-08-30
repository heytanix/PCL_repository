import gradio as gr
import torch
import torch.nn.functional as F
from torchvision.models.video import r3d_18
import cv2
import numpy as np
import threading
import queue
import time
from collections import deque

# Custom resize for videos (resizes each frame individually)
class ResizeVideo:
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        c, t, h, w = video.shape
        video_resized = torch.zeros((c, t, self.size, self.size), dtype=video.dtype)
        for i in range(t):
            frame = video[:, i, :, :].unsqueeze(0)  # (1, C, H, W)
            frame_resized = F.interpolate(frame, size=(self.size, self.size), mode='bilinear', align_corners=False)
            video_resized[:, i, :, :] = frame_resized.squeeze(0)
        return video_resized

# Custom normalize for videos
class NormalizeVideo:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1, 1)

    def __call__(self, video):
        return (video - self.mean) / self.std

transform = [
    ResizeVideo(112),
    NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
]

def apply_transforms(video):
    for t in transform:
        video = t(video)
    return video

# Global variables for streaming
streaming_active = False
current_model = None
current_device = None
frame_buffer = deque(maxlen=16)

# Load model
def load_model(model_path, device):
    model = r3d_18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Get available webcams
def get_available_webcams():
    available = []
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(str(i))
            cap.release()
    return available or ["0"]  # Default to 0 if none found

# Live prediction function
def predict_live_stream(image, model_path, device_choice, cam_index):
    global streaming_active, current_model, current_device, frame_buffer
    
    if image is None:
        return image
    
    try:
        # Initialize model if needed
        device = torch.device(device_choice)
        if current_model is None or current_device != device_choice:
            current_model = load_model(model_path, device)
            current_device = device_choice
        
        # Convert image to RGB and add to buffer
        if len(image.shape) == 3:
            rgb_frame = image
            frame_buffer.append(rgb_frame)
        
        # Default prediction
        prediction = "Waiting for frames..."
        confidence = 0.0
        color = (0, 255, 0)  # Green default
        
        # Make prediction when we have enough frames
        if len(frame_buffer) == 16:
            try:
                # Prepare frames for model
                frames = list(frame_buffer)
                frames_np = np.stack(frames, axis=0)  # (T, H, W, C)
                frames_np = frames_np.transpose((3, 0, 1, 2))  # (C, T, H, W)
                video_tensor = torch.from_numpy(frames_np).float() / 255.0
                
                # Apply transforms and predict
                video_tensor = apply_transforms(video_tensor)
                video_tensor = video_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = current_model(video_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    pred_idx = np.argmax(probs)
                    confidence = probs[pred_idx]
                    
                labels = ['Non-Violent', 'Violent']
                prediction = labels[pred_idx]
                color = (0, 255, 0) if pred_idx == 0 else (255, 0, 0)  # Green/Red
                
            except Exception as e:
                prediction = f"Model Error: {str(e)[:50]}"
                color = (255, 255, 0)  # Yellow for error
        
        # Draw overlay on image
        overlay_image = image.copy()
        h, w = overlay_image.shape[:2]
        
        # Draw bounding box
        margin = 50
        cv2.rectangle(overlay_image, (margin, margin), (w-margin, h-margin), color, 3)
        
        # Add prediction text with background
        text = f"{prediction} ({confidence:.2f})" if confidence > 0 else prediction
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x, text_y = margin + 10, margin + 40
        
        # Text background
        cv2.rectangle(overlay_image, (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        
        # Text
        cv2.putText(overlay_image, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return overlay_image
        
    except Exception as e:
        # Return original image with error message
        error_image = image.copy()
        cv2.putText(error_image, f"Error: {str(e)[:30]}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return error_image

# Gradio interface
with gr.Blocks(title="Live Violence Detection") as demo:
    gr.Markdown("# Live Real-Time Violence Detection")
    gr.Markdown("**Live streaming with instant predictions and overlays**")
    
    with gr.Row():
        with gr.Column():
            model_path = gr.Textbox(
                label="Model Path (.pth)", 
                value="violence_classifier.pth",
                placeholder="Path to your trained model"
            )
            device = gr.Dropdown(
                choices=["cpu", "cuda"], 
                label="Select Device", 
                value="cpu"
            )
            cam_index = gr.Dropdown(
                choices=get_available_webcams(), 
                label="Select Webcam",
                value="0"
            )
            
            gr.Markdown("### Instructions:")
            gr.Markdown("1. Set your model path and device")
            gr.Markdown("2. Select webcam (0=built-in, 1=external)")
            gr.Markdown("3. Allow camera access when prompted")
            gr.Markdown("4. See live predictions with colored overlays")
        
        with gr.Column():
            # Use Image component with webcam streaming - more reliable than Video
            webcam_stream = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="Live Feed with Detection"
            )
    
    # Connect the streaming function
    webcam_stream.stream(
        fn=predict_live_stream,
        inputs=[webcam_stream, model_path, device, cam_index],
        outputs=webcam_stream,
        stream_every=0.1,  # Process every 0.1 seconds for smooth real-time effect
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)