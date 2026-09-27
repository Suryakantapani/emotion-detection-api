from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import cv2

app = FastAPI(title="Emotion Detection API")
model = tf.keras.models.load_model("emotion_model.keras")
class_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"message": "Emotion Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    if len(faces) == 0:
        return {
            "error": "No face detected"
        }
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    face = image_array[y:y+h, x:x+w]
    face = Image.fromarray(face)
    face = face.resize((224, 224))
    img_array = np.array(face, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])
    return {
        "emotion": class_names[predicted_index],
        "confidence": round(confidence * 100, 2)
    }
