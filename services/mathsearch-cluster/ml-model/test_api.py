#!/usr/bin/env python3
"""
Simple test script for the YOLOv8 API
"""
import requests
import sys
import os

API_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"✅ Health check: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_predict(image_path):
    """Test the predict endpoint with an image file"""
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False
    
    print(f"Testing predict endpoint with {image_path}...")
    try:
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{API_URL}/predict",
                files={'file': f}
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Found {result['count']} detections")
            for i, box in enumerate(result['boxes']):
                print(f"   Box {i+1}: {box['class_name']} (confidence: {box['confidence']:.2f})")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 YOLOv8 API Test Script\n")
    
    # Test health
    if not test_health():
        print("\n❌ API server might not be running. Start it with: ./run_api.sh")
        sys.exit(1)
    
    # Test predict if image provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_predict(image_path)
    else:
        print("\n💡 To test prediction, provide an image path:")
        print("   python3 test_api.py path/to/image.png")

