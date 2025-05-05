from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np
import cv2
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import concurrent.futures

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("yolo-server/cardmg.pt")
executor = concurrent.futures.ThreadPoolExecutor()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>Live YOLO Detection</title></head>
    <body>
        <h2>Live YOLOv8 Detection</h2>
        <video id="video" width="640" height="480" autoplay></video>
        <canvas id="canvas" width="640" height="480" style="position:absolute; top:0; left:0;"></canvas>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');

            navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
                .then(stream => { video.srcObject = stream; })
                .catch(err => { alert('Error accessing camera: ' + err.message); });

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
                    .then(res => {
                        if (!res.ok) throw new Error("Server error: " + res.status);
                        return res.json();
                    })
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
                    .catch(error => console.error("Error in detection:", error));
                }, 'image/jpeg');
            }

            setInterval(sendFrame, 5000); // every 3s to reduce CPU
        </script>
    </body>
    </html>
    """

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((320, 240))

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(executor, model, image)

        detections = []
        boxes = results[0].boxes

        if boxes is not None:
            for box in boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = box
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": score,
                    "class_id": int(class_id),
                    "tag": model.names[int(class_id)]
                })

        return {"results": detections}
    except Exception as e:
        print("🔥 Error in /detect/:", e)
        return {"results": []}
