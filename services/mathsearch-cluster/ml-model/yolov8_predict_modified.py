from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train/weights/best.pt')
print("Loaded YOLO model!")

def predict_image(img_bgr):

    resized = cv2.resize(img_bgr, (640, 640))

    results = model.predict(resized, verbose=False)[0]

    boxes_out = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        label = int(box.cls[0])
        boxes_out.append([x1, y1, x2, y2, score, label])

    return {"boxes": boxes_out}

if __name__ == "__main__":
    import os

    # Test image path (customize)
    test_img_path = "test.png"

    if not os.path.exists(test_img_path):
        print(f"ERROR: test image '{test_img_path}' not found.")
    else:
        img = cv2.imread(test_img_path)
        out = predict_image(img)
        print("YOLO output:", out)
