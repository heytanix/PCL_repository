# Import necessary libraries
import gradio as gr  # For building web interface
import cv2  # OpenCV for image/video processing
import tempfile  # For creating temporary files
from ultralytics import YOLOv10  # YOLOv10 model implementation
import threading  # For handling webcam stream in background
import time  # For FPS calculation and timing
import queue  # For frame queue between threads

# Function to perform YOLOv10 inference on images or videos
def yolov10_inference(image, video, model_id, image_size, conf_threshold):
    # Load the YOLOv10 model from HuggingFace hub
    model = YOLOv10.from_pretrained(f'jameslahm/{model_id}')
    
    # If image input is provided
    if image:
        # Run model prediction on the image
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        # Get annotated image with bounding boxes
        annotated_image = results[0].plot()
        # Return image (convert BGR to RGB) and None for video
        return annotated_image[:, :, ::-1], None
    else:
        # Create temporary file for video processing
        video_path = tempfile.mktemp(suffix=".webm")
        # Write video data to temporary file
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        # Open video file for reading
        cap = cv2.VideoCapture(video_path)
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video file
        output_video_path = tempfile.mktemp(suffix=".webm")
        # Initialize video writer with VP8 codec
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run model prediction on each frame
            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            # Get annotated frame with bounding boxes
            annotated_frame = results[0].plot()
            # Write annotated frame to output video
            out.write(annotated_frame)

        # Release video resources
        cap.release()
        out.release()

        # Return None for image and path to annotated video
        return None, output_video_path

# Global variables for webcam streaming
webcam_active = False
result_queue = queue.Queue(maxsize=1)  # Queue to hold the latest processed frame
webcam_thread = None  # Thread for capturing webcam frames

# Function to capture frames from webcam
def webcam_capture_thread(webcam_id, model_id, image_size, conf_threshold):
    global webcam_active
    
    # Load the YOLOv10 model
    model = YOLOv10.from_pretrained(f'jameslahm/{model_id}')
    
    # Initialize webcam
    cap = cv2.VideoCapture(int(webcam_id))
    if not cap.isOpened():
        print(f"Error: Could not open webcam {webcam_id}")
        webcam_active = False
        return
    
    # Set webcam properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Variables for FPS calculation
    fps_count = 0
    fps_start_time = time.time()
    fps = 0
    
    # Main capture loop
    while webcam_active:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam")
            break
        
        # Update FPS calculation
        fps_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_start_time = time.time()
            
        try:
            # Run YOLOv10 prediction on the frame
            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            # Get annotated frame with bounding boxes
            annotated_frame = results[0].plot()
            
            # Add FPS counter to the frame
            cv2.putText(annotated_frame, f"FPS: {fps}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert BGR to RGB for Gradio display
            rgb_frame = annotated_frame[:, :, ::-1]
            
            # Put the processed frame in the result queue (overwrite old frame if queue is full)
            if result_queue.full():
                result_queue.get_nowait()
            result_queue.put(rgb_frame)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    # Clean up
    cap.release()
    print("Webcam capture thread stopped")

# Function to start webcam detection
def start_webcam_detection(webcam_id, model_id, image_size, conf_threshold):
    global webcam_active, webcam_thread
    
    # Stop any existing webcam thread
    stop_webcam_detection()
    
    # Clear queue
    while not result_queue.empty():
        result_queue.get()
    
    # Start new webcam thread
    webcam_active = True
    webcam_thread = threading.Thread(
        target=webcam_capture_thread,
        args=(webcam_id, model_id, image_size, conf_threshold),
        daemon=True
    )
    webcam_thread.start()
    
    # Wait a bit for the first frame
    time.sleep(1)
    
    # Get the first frame if available
    if not result_queue.empty():
        return result_queue.get()
    
    # Return a placeholder if no frame is available yet
    import numpy as np
    return np.zeros((480, 640, 3), dtype=np.uint8)

# Function to stop webcam detection
def stop_webcam_detection():
    global webcam_active, webcam_thread
    
    # Stop the webcam thread if it's running
    if webcam_active:
        webcam_active = False
        if webcam_thread:
            # Give the thread time to clean up
            time.sleep(0.5)
            webcam_thread = None
        return "Webcam detection stopped"
    return "Webcam detection is not running"

# Function to get the latest processed frame for the webcam feed
def update_webcam_output():
    # Return the latest processed frame if available
    if webcam_active and not result_queue.empty():
        return result_queue.get()
    
    # Return None if webcam is not active or no frame is available
    import numpy as np
    return np.zeros((480, 640, 3), dtype=np.uint8)

# Helper function for example processing (image only)
def yolov10_inference_for_examples(image, model_path, image_size, conf_threshold):
    # Call main inference function for example images
    annotated_image, _ = yolov10_inference(image, None, model_path, image_size, conf_threshold)
    return annotated_image

# Main application function
def app():
    # Create Gradio interface using Blocks API
    with gr.Blocks() as blocks:
        # Layout with two columns (input and output)
        with gr.Row():
            with gr.Column():
                # Input image component
                image = gr.Image(type="pil", label="Image", visible=True)
                # Input video component
                video = gr.Video(label="Video", visible=False)
                # Webcam ID input (for live detection)
                webcam_id = gr.Number(value=0, label="Webcam ID (usually 0 for default camera)", visible=False, precision=0)
                # Radio button to select input type
                input_type = gr.Radio(
                    choices=["Image", "Video", "Live Webcam"],
                    value="Image",
                    label="Input Type",
                )
                # Dropdown to select YOLOv10 model variant
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov10n",
                        "yolov10s",
                        "yolov10m",
                        "yolov10b",
                        "yolov10l",
                        "yolov10x",
                    ],
                    value="yolov10m",
                )
                # Slider to select image size for inference
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                # Slider to set confidence threshold
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                # Button for image/video detection
                yolov10_infer = gr.Button(value="Detect Objects")
                
                # Buttons for webcam control
                with gr.Row(visible=False) as webcam_controls:
                    start_webcam = gr.Button(value="Start Webcam Detection")
                    stop_webcam = gr.Button(value="Stop Webcam Detection")
                    refresh_webcam = gr.Button(value="Refresh Frame")

            # Output column
            with gr.Column():
                # Output image component
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                # Output video component
                output_video = gr.Video(label="Annotated Video", visible=False)
                # Live webcam output
                webcam_output = gr.Image(type="numpy", label="Live Detection", visible=False)

        # Function to update UI visibility based on input type
        def update_visibility(input_type):
            if input_type == "Image":
                # Show image components, hide others
                return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), 
                        gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=True))
            elif input_type == "Video":
                # Show video components, hide others
                return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                        gr.update(visible=False), gr.update(visible=True))
            else:  # Live Webcam
                # Show webcam components, hide others
                return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=True), gr.update(visible=False))

        # Connect input type change to visibility update
        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, webcam_id, webcam_controls, output_image, output_video, webcam_output, yolov10_infer],
        )

        # Wrapper function for running inference
        def run_inference(image, video, model_id, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return yolov10_inference(image, None, model_id, image_size, conf_threshold)
            else:
                return yolov10_inference(None, video, model_id, image_size, conf_threshold)

        # Connect inference button to processing function
        yolov10_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )

        # Connect webcam control buttons
        start_webcam.click(
            fn=start_webcam_detection,
            inputs=[webcam_id, model_id, image_size, conf_threshold],
            outputs=[webcam_output],
        )
        
        stop_webcam.click(
            fn=stop_webcam_detection,
            inputs=[],
            outputs=[],
        )
        
        # Connect refresh button to update the webcam frame
        refresh_webcam.click(
            fn=update_webcam_output,
            inputs=[],
            outputs=[webcam_output],
        )

        # Example section with pre-loaded images
        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    "yolov10s",
                    640,
                    0.25,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    "yolov10s",
                    640,
                    0.25,
                ],
            ],
            fn=yolov10_inference_for_examples,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
            cache_examples='lazy',  # Cache examples for better performance
        )

# Create main Gradio application
gradio_app = gr.Blocks()
with gradio_app:
    # Add title HTML
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv10: Real-Time End-to-End Object Detection
    </h1>
    """)
    # Add links to paper and GitHub
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2405.14458' target='_blank'>arXiv</a> | <a href='https://github.com/THU-MIG/yolov10' target='_blank'>github</a>
        </h3>
        """)
    # Layout row for the main application
    with gr.Row():
        with gr.Column():
            # Include the app function's content
            app()

# Main entry point
if __name__ == '__main__':
    # Launch the Gradio application
    gradio_app.launch()