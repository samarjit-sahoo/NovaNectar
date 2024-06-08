from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the FaceNet model
model = load_model('facenet_keras.h5')

def preprocess_image(image):
    image = Image.fromarray(image)
    image = image.resize((160, 160))
    image = np.asarray(image)
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0)
    return image

def get_embedding(face):
    face = preprocess_image(face)
    embedding = model.predict(face)
    return embedding
