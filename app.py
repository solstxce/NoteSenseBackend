from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List
import io

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the YOLOv8 model (do this only once when starting the server)
model = YOLO('best.pt')  # Replace with your model path

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run inference
    results = model(img, conf=0.25)  # Adjust confidence threshold as needed
    
    # Process results
    result = results[0]
    predictions = []
    
    for box in result.boxes:
        pred = {
            "bbox": box.xyxy[0].tolist(),  # Convert bbox to list
            "confidence": float(box.conf),
            "class": int(box.cls),
            "class_name": result.names[int(box.cls)]
        }
        predictions.append(pred)
    
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 