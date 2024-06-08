import cv2

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


# Example usage
# if __name__ == "__main__":
#     image = cv2.imread('C:/Users/samar/Random/test1.png')
#     faces = detect_faces(image)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     cv2.imshow('Faces', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
