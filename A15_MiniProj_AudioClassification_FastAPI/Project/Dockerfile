FROM python:3.12-alpine

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY audio_classification_spectrogram.onnx .

COPY main.py .

CMD ["python", "main.py"]
