# ResNet3D18 CNN Real-Time Human Violence Detection

## Project Overview
This deep learning project focuses on building a portable software solution for real-time human violence detection. Leveraging the ResNet3D18 convolutional neural network, the system is designed to recognize violent and non-violent actions from video streams. It supports multi-device integration, enabling deployment on both CPU and GPU environments for broad accessibility. The software aims to automatically detect incidents of violence and promptly alert relevant authorities, enhancing safety and response times.  
*(Last updated: September 3rd, 2025)*

## Architecture Diagrams of models
- of the ResNet-18 CNN Model
![ResNet-18 Architecture Diagram](/Assets/Images/ResNet_18_architecture.png)

## Features
- Portable (CPU Model, CUDA Model)
   - Designed for broad compatibility: includes a CPU-optimized model for devices without dedicated GPUs.
- Automated Distress Signals
- Violence Detection
- Non-Violence Detection (For differentiation)

## Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- PyTorch (`torch`, `torchvision`)
- Gradio
- Decord (`decord`)
- NumPy
- scikit-learn (`scikit-learn`)
- Matplotlib (`matplotlib`)
- Threading and Queue modules (standard library)
- datetime module (standard library)
- os module (standard library)
- tqdm (optional, for progress bars)

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