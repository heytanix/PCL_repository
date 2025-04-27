# YOLOv10 Real-Time Object Detection

## Project Overview
This project enhances the YOLOv10 object detection system with real-time webcam integration for surveillance applications.
The implementation provides a user-friendly Gradio interface for object detection across images, videos, and live webcam feeds, making it suitable for deployment in drone-based surveillance systems for detecting violence or suspicious activities.

## Features
- Multi-modal object detection (images, videos, and real-time webcam)
- Support for all YOLOv10 model variants (nano to extra large)
- Real-time FPS counter for performance monitoring
- Adjustable detection parameters (confidence threshold, image size)
- Threaded architecture for smooth webcam performance
- Pre-loaded example images for demonstration

## Requirements
- Python 3.8+
- OpenCV
- Gradio
- Ultralytics YOLOv10
- Threading and Queue modules (standard library)

## Installation
```bash
# Clone the repository
git clone https://github.com/your-username/yolov10-detection.git
cd yolov10-detection

# Install dependencies
pip install ultralytics gradio opencv-python
```

## Usage
Run the application with:
```bash
python app.py
```

The web interface will be accessible at `http://localhost:7860` by default.

### Interface Options:
1. **Input Type**: Choose between Image, Video, or Live Webcam
2. **Model Selection**: Select from YOLOv10 variants (n, s, m, b, l, x)
3. **Image Size**: Adjust the input resolution (higher = more accurate but slower)
4. **Confidence Threshold**: Set detection confidence level (0.0-1.0)
5. **Webcam ID**: Select camera device (usually 0 for default webcam)

## Modifications from Base YOLOv10
The following enhancements have been made to the original YOLOv10 implementation:

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
- Thanish Chinnappa KC
- Sujeeth RK
- Tanisha Vernekar
- Tejas RU
- S Uday Gowda

## License
This project builds upon the YOLOv10 implementation, which is available under its original license. Our modifications are provided for research and academic purposes. When using or distributing this software, please acknowledge both the original YOLOv10 work and our modifications.

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
- [YOLOv10 GitHub Repository](https://github.com/THU-MIG/yolov10)
- [YOLOv10 Paper](https://arxiv.org/abs/2405.14458)
