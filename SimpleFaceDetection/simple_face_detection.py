#!/usr/bin/python

# Simple face detection example with OpenCV
# Requires Python 3 and OpenCV 3

import cv2


# Load the sample image
image = cv2.imread('data/abba.jpg')

# Load the OpenCV classifier to detect faces
faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = faceCascade.detectMultiScale(
	image,
	scaleFactor=1.2,
	minNeighbors=5,
	minSize=(50, 50)
)

# The faces variable now contains an array of Nx4 elements where N is the number faces detected

print("Found", len(faces), "faces")

# Draw a rectangle around each face
for (x, y, w, h) in faces:
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image
cv2.imshow("Faces found", image)

# Wait for any key before exiting
cv2.waitKey(0)
