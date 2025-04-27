# Import necessary libraries
import gradio as gr  # For building web interface
import cv2  # OpenCV for image/video processing
import tempfile  # For creating temporary files
from ultralytics import YOLOv10  # YOLOv10 model implementation

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

# Helper function for example processing (image only)
def yolov10_inference_for_examples(image, model_path, image_size, conf_threshold):
    # Call main inference function for example images
    annotated_image, _ = yolov10_inference(image, None, model_path, image_size, conf_threshold)
    return annotated_image

# Main application function
def app():
    # Create Gradio interface using Blocks API
    with gr.Blocks():
        # Layout with two columns (input and output)
        with gr.Row():
            with gr.Column():
                # Input image component
                image = gr.Image(type="pil", label="Image", visible=True)
                # Input video component
                video = gr.Video(label="Video", visible=False)
                # Radio button to select input type
                input_type = gr.Radio(
                    choices=["Image", "Video"],
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
                # Button to trigger inference
                yolov10_infer = gr.Button(value="Detect Objects")

            # Output column
            with gr.Column():
                # Output image component
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                # Output video component
                output_video = gr.Video(label="Annotated Video", visible=False)

        # Function to update UI visibility based on input type
        def update_visibility(input_type):
            # Show/hide components based on selected input type
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)

            return image, video, output_image, output_video

        # Connect input type change to visibility update
        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
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
