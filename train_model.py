import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
from get_embedding import get_embedding

def load_faces(directory):
    faces = []
    labels = []
    for label in os.listdir(directory):
        path = os.path.join(directory, label)
        if os.path.isdir(path):
            for filename in os.listdir(path):
                filepath = os.path.join(path, filename)
                face = cv2.imread(filepath)
                if face is not None:
                    embedding = get_embedding(face)
                    faces.append(embedding)
                    labels.append(label)
                else:
                    print(f"Failed to read {filepath}")
    return np.array(faces), np.array(labels)

# Load dataset
faces, labels = load_faces('dataset')

# Debug: Print the shape of the faces and labels arrays
print(f"Faces shape: {faces.shape}")
print(f"Labels shape: {labels.shape}")

# Flatten the embeddings
faces = faces.reshape((faces.shape[0], faces.shape[2]))

# Debug: Print the new shape of the faces array
print(f"Flattened faces shape: {faces.shape}")

# Encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Debug: Print the classes found by the encoder
print(f"Classes: {encoder.classes_}")

# Train model
clf = SVC(kernel='linear', probability=True)
clf.fit(faces, encoded_labels)

# Save the model
with open('face_recognition_model.pkl', 'wb') as f:
    pickle.dump((encoder, clf), f)

print("Model training complete. Model saved as face_recognition_model.pkl")
