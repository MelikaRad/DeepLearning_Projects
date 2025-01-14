#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install fastapi uvicorn  python-multipart')

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Optional

app = FastAPI()

model_path = 'drive/MyDrive/MyModels/PersianDigitClassification_FastAPI.keras'
model = keras.models.load_model(model_path)

def predict_image(image_path):
    img = Image.open(image_path).convert('L')  # convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array[..., np.newaxis]
    img_array = img_array / 255.0
    predictions = model.predict(img_array[np.newaxis, ...])  # add batch dim
    predicted_number = np.argmax(predictions)
    return predicted_number

@app.post("/predict_number/")
async def predict_number(image: UploadFile = File(...)):
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(image.file.read())
    predicted_number = predict_image(image_path)
    
    return JSONResponse(content={"predicted_number": int(predicted_number)}, media_type="application/json")