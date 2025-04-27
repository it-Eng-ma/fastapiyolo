from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np
import cv2

app = FastAPI()

# Load YOLO model
model = YOLO("yolov8n.pt")

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # Read the incoming frame as bytes
    image_bytes = await file.read()
    # Convert bytes to an image using PIL
    image = Image.open(io.BytesIO(image_bytes))

    # Perform inference with YOLO
    results = model(image)  # Inference on the frame

    # Extract detections
    detections = []
    for box in results[0].boxes.data.tolist():  # Loop through the boxes
        x1, y1, x2, y2, score, class_id = box
        detections.append({
            "box": [x1, y1, x2, y2],    # Bounding box coordinates
            "confidence": score,         # Confidence score
            "class_id": int(class_id),   # Class ID
            "tag": model.names[int(class_id)]  # Class name
        })

    # Return detections as JSON
    return {"results": detections}
