#!/bin/bash
# Script to run the YOLOv8 inference API server

# Change to the script directory
cd "$(dirname "$0")"

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -q -r requirements_api.txt

# Check if model file exists
if [ ! -f "runs/detect/train/weights/best.pt" ]; then
    echo "⚠️  WARNING: Model file not found at runs/detect/train/weights/best.pt"
    echo "Please ensure the model is trained and available at that path"
    exit 1
fi

# Run the API server
echo "🚀 Starting YOLOv8 Inference API server..."
echo "📡 API will be available at: http://0.0.0.0:8000"
echo "📚 API docs will be available at: http://0.0.0.0:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn api:app --host 0.0.0.0 --port 8000 --reload

