from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np
import cv2
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow frontend requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this to specific origins)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load YOLO model
model = YOLO("yolov8n.pt")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Live YOLO Detection</title>
    </head>
    <body>
        <h2>Live YOLOv8 Detection</h2>
        <video id="video" width="640" height="480" autoplay></video>
        <canvas id="canvas" width="640" height="480" style="position:absolute; top:0; left:0;"></canvas>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');

            // Request access to the rear camera (for mobile devices)
            navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
                .then(stream => {
                    video.srcObject = stream;
                })
                .catch(err => {
                    alert('Error accessing camera: ' + err.message);  // Show error message if camera access fails
                });

            function sendFrame() {
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = 640;
                tempCanvas.height = 480;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(video, 0, 0, 640, 480);
                tempCanvas.toBlob(blob => {
                    const formData = new FormData();
                    formData.append('file', blob, 'frame.jpg');

                    fetch('/detect/', {
                        method: 'POST',
                        body: formData
                    })
                    .then(res => res.json())
                    .then(data => {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        for (const det of data.results) {
                            const [x1, y1, x2, y2] = det.box;
                            ctx.strokeStyle = "red";
                            ctx.lineWidth = 2;
                            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                            ctx.font = "16px Arial";
                            ctx.fillStyle = "red";
                            ctx.fillText(det.tag + " " + (det.confidence*100).toFixed(1) + "%", x1, y1 - 5);
                        }
                    })
                    .catch(error => console.error("Error in detection:", error));  // Log any errors from the fetch request
                }, 'image/jpeg');
            }

            setInterval(sendFrame, 1000); // Send 1 frame per second
        </script>
    </body>
    </html>
    """

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