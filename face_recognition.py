import cv2
import os

# Read picture
image_path = os.path.join(os.getcwd(), 'image.jpg')
image = cv2.imread(image_path)

# Convert picture to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialisierung des Cascade Classifiers zur Gesichtserkennung
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

# face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# draw a green (0,255,0) rectangle around the detected faces (when showing the pic) 
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show picture
cv2.imshow('Gesichtserkennung', image)
cv2.waitKey(0)

# save the detected faces as new image 
for i, (x, y, w, h) in enumerate(faces):
    face_image = image[y:y+h, x:x+w]
    face_path = os.path.join(os.getcwd(), f'face{i}.jpg')
    cv2.imwrite(face_path, face_image)
