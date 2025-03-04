# -*- coding: utf-8 -*-

from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
import numpy as np
import librosa
import tensorflow as tf
import uvicorn

app = FastAPI()

ort_session = ort.InferenceSession("audio_classification_spectrogram.onnx")

def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def audio_to_spectrogram(file_path):
    audio, sr = librosa.load(file_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if len(audio) < 2 * 22050:
        audio = np.pad(audio, (0, 2 * 22050 - len(audio)), mode='constant')
    elif len(audio) > 2 * 22050:
        audio = audio[:2 * 22050]

    spectrogram = get_spectrogram(audio)
    resized_spectrogram = tf.image.resize(spectrogram, (32, 32))
    return resized_spectrogram.numpy()

@app.post("/classify_audio/")
async def classify_audio(file: UploadFile = File(...)):
    # save uploaded file temporarily
    with open("temp_audio.wav", "wb") as buffer:
        buffer.write(await file.read())

    # process audio file
    spectrogram = audio_to_spectrogram("temp_audio.wav")
    spectrogram = np.expand_dims(spectrogram, axis=0) # for batch dim

    # run inference
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    prediction = ort_session.run([output_name], {input_name: spectrogram.astype(np.float32)})[0]

    class_index = np.argmax(prediction)
    return {"class": int(class_index)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)