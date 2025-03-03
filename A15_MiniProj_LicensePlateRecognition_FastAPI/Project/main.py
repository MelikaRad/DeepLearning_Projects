# -*- coding: utf-8 -*-
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
import tensorflow as tf
from tensorflow.keras import layers
import io

app = FastAPI()

# Load your models and initialize necessary components
model = YOLO('plate_keypoints_detector.pt')
ort_session = ort.InferenceSession("license_plate_recognition.onnx")

input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'H', 'J', 'K', 'L', 'M', 'N', 'S', 'T', 'V', 'X', 'Y', 'Z']
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def process_plate(plate):
    plate = cv2.resize(plate, (256, 65))
    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    plate = np.expand_dims(plate, -1)
    plate = plate.transpose(1, 0, 2)  # HWC to CHW
    plate = np.expand_dims(plate, 0).astype(np.float32) / 255.0
    return plate

def recognize_plate(plate):
    preds = ort_session.run([output_name], {input_name: plate})[0]
    preds = preds.argmax(axis=-1)
    pred_texts = []
    for p in preds:
        label = tf.strings.reduce_join(num_to_char(p)).numpy().decode("utf-8")
        pred_texts.append(label)
    return pred_texts[0]

@app.post("/recognize_plate")
async def recognize_license_plate(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)

    for result in results:
        keyps = result.keypoints.xy.cpu().numpy().astype('int32')[0]

    plate_corners = np.float32([
        [keyps[0,0],keyps[0,1]-5],  # Top-left
        [keyps[1,0],keyps[1,1]-5],  # Top-right
        [keyps[2,0],keyps[2,1]+5],  # Bottom-right
        [keyps[3,0],keyps[3,1]+5]   # Bottom-left
    ])

    dst_points = np.float32([
        [0, 0],           # Top-left
        [256 -1, 0],     # Top-right
        [256 - 1, 65 - 1], # Bottom-right
        [0, 65 - 1]       # Bottom-left
    ])

    M = cv2.getPerspectiveTransform(plate_corners, dst_points)
    transformed_plate = cv2.warpPerspective(img, M, (256, 65))

    plate = process_plate(transformed_plate)
    recognized_text = recognize_plate(plate)

    return JSONResponse(content={"plate_number": recognized_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
