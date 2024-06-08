import cv2
import pickle
from detect_faces import detect_faces
from get_embedding import get_embedding
from tensorflow.keras.models import load_model

# Load the FaceNet model
model = load_model('facenet_keras.h5')

# Load the pre-trained model and the encoder
model_file = 'face_recognition_model.pkl'

if not os.path.exists(model_file):
    print(f"Model file {model_file} does not exist. Please run train_model.py first.")
    exit(1)

with open(model_file, 'rb') as f:
    encoder, clf = pickle.load(f)

def recognize_face(image):
    faces = detect_faces(image)
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        embedding = get_embedding(face)
        embedding = embedding.reshape(1, -1)
        prediction = clf.predict(embedding)
        label = encoder.inverse_transform(prediction)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, label[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image

# Example usage
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = recognize_face(frame)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
