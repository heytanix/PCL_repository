import gradio as gr
import torch
import torch.nn.functional as F
from torchvision.models.video import r3d_18
import cv2
import numpy as np
import time
import os

# Custom resize for videos (resizes each frame individually)
class ResizeVideo:
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        # video: (C, T, H, W)
        c, t, h, w = video.shape
        video_resized = torch.zeros((c, t, self.size, self.size), dtype=video.dtype)
        for i in range(t):
            frame = video[:, i, :, :].unsqueeze(0)  # (1, C, H, W)
            frame_resized = F.interpolate(frame, size=(self.size, self.size), mode='bilinear', align_corners=False)
            video_resized[:, i, :, :] = frame_resized.squeeze(0)
        return video_resized

# Custom normalize for videos (broadcasts mean/std across T, H, W)
class NormalizeVideo:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1, 1)

    def __call__(self, video):
        # video: (C, T, H, W)
        return (video - self.mean) / self.std

# Compose the transforms
transform = [
    ResizeVideo(112),  # Resize to 112x112 per frame
    NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])  # Normalize across all frames
]

# Function to apply transforms sequentially
def apply_transforms(video):
    for t in transform:
        video = t(video)
    return video

# Function to capture video from webcam and process into tensor
def capture_and_process(cam_index, duration_sec=5, clip_len=16):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with index {cam_index}")

    frames = []
    start_time = time.time()

    while (time.time() - start_time) < duration_sec:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames captured from webcam.")

    # Sample or pad to clip_len frames
    if len(frames) >= clip_len:
        indices = np.linspace(0, len(frames) - 1, clip_len, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        frames = frames * (clip_len // len(frames) + 1)
        frames = frames[:clip_len]

    # Convert to tensor (C, T, H, W)
    frames_np = np.stack(frames, axis=0)  # (T, H, W, C)
    frames_np = frames_np.transpose((3, 0, 1, 2))  # (C, T, H, W)
    video_tensor = torch.from_numpy(frames_np).float() / 255.0  # Normalize to [0,1]

    # Apply custom transforms
    video_tensor = apply_transforms(video_tensor)

    # Add batch dimension (1, C, T, H, W)
    video_tensor = video_tensor.unsqueeze(0)
    return video_tensor

# Load the model
def load_model(model_path, device):
    model = r3d_18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary: non-violent, violent
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Prediction function
def predict_action(model_path, device_choice, cam_index):
    device = torch.device(device_choice)
    model = load_model(model_path, device)

    try:
        video_tensor = capture_and_process(int(cam_index))
        video_tensor = video_tensor.to(device)
        with torch.no_grad():
            outputs = model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    except Exception as e:
        return f"Error: {str(e)}"

    labels = ['Non-Violent', 'Violent']
    pred_idx = probabilities.argmax()
    confidence = probabilities[pred_idx]
    return f"Prediction: {labels[pred_idx]} (Confidence: {confidence:.2f})"

# Detect available webcams (check up to 5 indices)
def get_available_webcams(max_test=5):
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(str(i))
            cap.release()
    return available

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Violence Detection Model Tester")
    
    model_path = gr.Textbox(label="Model Path (.pth)", value="violence_classifier.pth")
    device = gr.Dropdown(choices=["cpu", "cuda"], label="Select Device", value="cpu")
    cam_index = gr.Dropdown(choices=get_available_webcams(), label="Select Webcam (0: Built-in, 1: External, etc.)")
    
    output = gr.Textbox(label="Prediction")
    button = gr.Button("Run Prediction")
    
    button.click(predict_action, inputs=[model_path, device, cam_index], outputs=output)

demo.launch(share=True)