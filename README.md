# ResNet3D18 CNN Real-Time Human Violence Detection

## Project Overview
- This project is dedicated to developing a portable software that supports multi-device integration to detect violence and alert the respective authorities.
- ResNet3D18: For Violence/Non-Violence recognition
- (Last Update: August 31-2025)

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
- Thanish Chinnappa KC (Lead Developer)
- Sujeeth RK (Writer: Research Paper)
- Tanisha Vernekar (Writer: Documentation)
- Tejas RU (Co-Developer)
- S Uday Gowda (Data Quality Analyst)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation
If you use this work in your research or project, please cite:

```
@article{yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={YOLOv10 Authors},
  journal={arXiv preprint arXiv:2405.14458},
  year={2025}
}
```

## References
- [YOLOv11 GitHub Repository]([https://github.com/THU-MIG/yolov10](https://github.com/ultralytics/ultralytics))

- [YOLOv11 Paper]([https://arxiv.org/abs/2405.14458](https://www.arxiv.org/abs/2410.17725))


