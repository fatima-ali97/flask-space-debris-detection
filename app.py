from flask import Flask, render_template, request, jsonify
import os
from ultralytics import YOLO
import cv2


app = Flask(__name__)

# Paths
MODEL_PATH = 'best.pt'
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model once at the beginning
model = YOLO(MODEL_PATH)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=["POST","GET"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Run YOLO model
    results = model(file_path)[0]
    conf_threshold = 0.6

    # Read uploaded image
    img = cv2.imread(file_path)

    # Loop over detections
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {conf:.2f}"
            # Draw box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save result image
    result_filename = f"result_{file.filename}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, img)

    print(f"Saved result to {result_path}")
    result_filename = f"result_{file.filename}"
    result_img = f"results/{result_filename}"   # relative to static
    original_file_path = f"uploads/{file.filename}"
    return render_template("index.html", result_img=result_img, upload_img=original_file_path)


if __name__ == '__main__':
    app.run(debug=True)
