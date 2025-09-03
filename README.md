# ResNet3D18 CNN Real-Time Human Violence Detection

## Attention Collaborators
**Important Notice:**  
This repository is now private to maintain confidentiality. Please do not share any code, data, or documentation from this project with anyone outside the team. The project is no longer open source and access is restricted to collaborators only. These measures have been taken to protect our work from unauthorized use,
Specifically in response to concerns regarding Dr. Sagar KG. (Sagar KG wanted to sell us out, make money on our work and rob our credits)
*(Integrity and unity is demanded in the current situation, Kindly cooperate)*

## Project Overview
This deep learning project focuses on building a portable software solution for real-time human violence detection. Leveraging the ResNet3D18 convolutional neural network, the system is designed to recognize violent and non-violent actions from video streams. It supports multi-device integration, enabling deployment on both CPU and GPU environments for broad accessibility. The software aims to automatically detect incidents of violence and promptly alert relevant authorities, enhancing safety and response times.  
*(Last updated: September 3rd, 2025)*

## Architecture Diagrams of models
**ResNet-18 Convolutional Neural Network(CNN) Model**
![ResNet-18 Architecture Diagram](/Assets/Images/ResNet_18_architecture.png)

**Vision Language Model (VLM) Architecture**
![VLM Architecture Diagram](/Assets/Images/VLM_Architecture.png)

## Features
- Portable (CPU Model, CUDA Model)
   - Designed for broad compatibility: includes a CPU-optimized model for devices without dedicated GPUs.
- Automated Distress Signals
- Violence Detection
- Non-Violence Detection (For differentiation)

## Requirements
- Python 3.8+
- OpenCV (`opencv-python>=4.9.0.80`)
- PyTorch (`torch>=2.3.0`, `torchvision>=0.18.0`)
- Gradio (`gradio>=4.26.0`)
- Decord (`decord`)
- NumPy (`numpy>=1.26.4`)
- scikit-learn (`scikit-learn`)
- Matplotlib (`matplotlib`)
- Threading and Queue modules (standard library)
- datetime module (standard library)
- os module (standard library)
- tqdm (optional, for progress bars)
- onnx
- onnxruntime
- pycocotools
- PyYAML
- scipy
- onnxslim
- onnxruntime-gpu
- psutil
- py-cpuinfo
- huggingface-hub
- safetensors

## Installation (Collaborator only)
```bash
# Clone the repository (Only possible if you're a collaborator)
gh repo clone heytanix/PCL_repository
cd PCL_repository

# Install dependencies
pip install requirements.txt
```

## Usage
- For training the ResNet-18 CNN model
   - Use 3d_cnn_with_videos.ipynb **This notebook trains 3D CNN from videos directly**
- For testing the ResNet-18 Based CNN model
   - Use tester.py **This script allows to launch a gradio based webapp to test the said model**
   - The web interface will be accessible at `http://127.0.0.1:7860` by default.

### Interface Options:
1. **Input Type**: Live camera input (Stable as of August 31-2025)
2. **Device Selection**: tester.py allows to choose between CPU/CUDA (as of August 31, 2025)
3. **Webcam ID**: Select camera device (usually 0 for default webcam)

1. **Real-time webcam integration**:
   - Added threaded architecture for concurrent frame capture and processing
   - Implemented frame queue system for smooth display
   - Added FPS counter for performance monitoring

2. **UI Improvements**:
   - Added webcam input option with dedicated controls
   - Dynamic interface that adapts based on the selected input type
   - Added webcam device selection for multi-camera setups

3. **Performance Optimizations**:
   - Frame queue implementation to prevent UI blocking
   - Background processing to maintain responsiveness
   - Proper resource cleanup for memory management

## Project Team
See the [CONTRIBUTING](CONTRIBUTING.md) file for the list of project contributors.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.