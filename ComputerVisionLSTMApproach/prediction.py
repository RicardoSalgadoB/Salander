import os
# Correct the paths with either double backslashes or raw string literals
c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
a = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib"
b = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include"
gstreamerPath = r"C:\gstreamer\1.0\msvc_x86_64\bin"

# Adding the required directories to the DLL search path
os.add_dll_directory(gstreamerPath)
os.add_dll_directory(c)
os.add_dll_directory(a)
os.add_dll_directory(b)
import cv2
import numpy as np
from ultralytics import YOLO
import subprocess

#Lin to realesrgan portable for not using cmake https://github.com/xinntao/Real-ESRGAN/releases

esrgan_cmd = f"realesrgan-ncnn-vulkan.exe -i test1.png -o test1_out.png"
process = subprocess.Popen(esrgan_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
output, error = process.communicate()

def process_image(image_path):
    model = YOLO("best.pt")  # Load your trained model
    model.to('cuda')  # Move the model to GPU and comment line if CUDA and Pytorch +cud is not installed
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error loading image: {image_path}")
        return
    frame_height, frame_width, _ = frame.shape

    results = model(frame)

    for result in results:
        boxes = result.boxes  # Get detected boxes

        for box in boxes:
            # Extract coordinates (xyxy format)
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy().astype(int)

            # Define the polygon points (rectangular in this case)
            pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], np.int32)
            pts = pts.reshape((-1, 1, 2))  # Reshape for OpenCV polylines

            # Draw the polygon (bounding box)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Add label and confidence score
            label = f"Class {int(box.cls[0])} {box.conf[0]:.2f}"  # Assuming box.cls contains the class and box.conf contains the confidence score
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image with bounding polygons
    cv2.imshow("Detections", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    image_path = "test1.png"  # Replace with your image file path
    process_image(image_path)
