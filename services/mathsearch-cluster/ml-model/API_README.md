# YOLOv8 Inference API

This API serves your trained YOLOv8 model as a REST endpoint for running inference on images.

## Quick Start

### 1. Run the API Server

```bash
cd /home/cadmin/computecluster/services/mathsearch-cluster/ml-model
./run_api.sh
```

Or manually:
```bash
cd /home/cadmin/computecluster/services/mathsearch-cluster/ml-model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_api.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 2. Access the API

- **API Base URL**: `http://10.49.7.37:8000` (or `http://localhost:8000` if accessing locally)
- **Interactive Docs**: `http://10.49.7.37:8000/docs` (Swagger UI)
- **Alternative Docs**: `http://10.49.7.37:8000/redoc` (ReDoc)

## API Endpoints

### Health Check
```bash
curl http://10.49.7.37:8000/health
```

### Predict (File Upload)
```bash
curl -X POST "http://10.49.7.37:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.png"
```

### Predict (Base64)
```bash
curl -X POST "http://10.49.7.37:8000/predict/base64" \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_string"}'
```

## Response Format

```json
{
  "boxes": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "class": 0,
      "class_name": "big-eqn"
    },
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.87,
      "class": 1,
      "class_name": "inline-eqn"
    }
  ],
  "count": 2
}
```

## Running as a Background Service

### Using screen (recommended for SSH sessions)
```bash
screen -S yolov8-api
cd /home/cadmin/computecluster/services/mathsearch-cluster/ml-model
./run_api.sh
# Press Ctrl+A then D to detach
```

To reattach:
```bash
screen -r yolov8-api
```

### Using tmux
```bash
tmux new -s yolov8-api
cd /home/cadmin/computecluster/services/mathsearch-cluster/ml-model
./run_api.sh
# Press Ctrl+B then D to detach
```

To reattach:
```bash
tmux attach -t yolov8-api
```

### Using systemd (for production)

Create `/etc/systemd/system/yolov8-api.service`:
```ini
[Unit]
Description=YOLOv8 Inference API
After=network.target

[Service]
Type=simple
User=cadmin
WorkingDirectory=/home/cadmin/computecluster/services/mathsearch-cluster/ml-model
Environment="PATH=/home/cadmin/computecluster/services/mathsearch-cluster/ml-model/venv/bin"
ExecStart=/home/cadmin/computecluster/services/mathsearch-cluster/ml-model/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable yolov8-api
sudo systemctl start yolov8-api
sudo systemctl status yolov8-api
```

## Python Client Example

```python
import requests

# Upload image file
with open('test_image.png', 'rb') as f:
    response = requests.post(
        'http://10.49.7.37:8000/predict',
        files={'file': f}
    )
    print(response.json())
```

## Testing

Test with a sample image:
```bash
# Make sure you have a test image
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test.png"
```

## Notes

- The model is loaded once at startup for better performance
- Images are automatically resized to 640x640 before inference
- The API supports CORS for web applications
- Model path: `runs/detect/train/weights/best.pt`

