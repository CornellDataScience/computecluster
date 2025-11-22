"""
YOLOv8 Model Inference API
Hosts the trained YOLOv8 model as a REST API endpoint
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os

# Initialize FastAPI app
app = FastAPI(title="YOLOv8 Inference API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_PATH = 'runs/detect/train/weights/best.pt'
model = None

@app.on_event("startup")
async def load_model():
    """Load the YOLOv8 model when the server starts"""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print(f"✅ YOLOv8 model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def predict_image(img_bgr):
    """Run inference on a BGR image"""
    if model is None:
        raise RuntimeError("Model not loaded")
    
    resized = cv2.resize(img_bgr, (640, 640))
    results = model.predict(resized, verbose=False)[0]
    
    boxes_out = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        label = int(box.cls[0])
        boxes_out.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": score,
            "class": label,
            "class_name": "big-eqn" if label == 0 else "inline-eqn" if label == 1 else f"class_{label}"
        })
    
    return {"boxes": boxes_out, "count": len(boxes_out)}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model": "YOLOv8",
        "model_path": MODEL_PATH,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Run inference on an uploaded image
    
    Args:
        file: Image file (jpg, png, etc.)
    
    Returns:
        JSON with detected boxes, confidence scores, and class labels
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        contents = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Run prediction
        result = predict_image(img_bgr)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict/base64")
async def predict_base64(data: dict):
    """
    Run inference on a base64 encoded image
    
    Args:
        data: JSON with "image" field containing base64 encoded image
    
    Returns:
        JSON with detected boxes, confidence scores, and class labels
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import base64
        
        if "image" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        # Decode base64 image
        image_data = data["image"]
        if image_data.startswith("data:image"):
            # Remove data URL prefix if present
            image_data = image_data.split(",")[1]
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode base64 image")
        
        # Run prediction
        result = predict_image(img_bgr)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

