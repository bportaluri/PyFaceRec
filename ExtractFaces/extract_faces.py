#!/usr/bin/python

# Simple face detection example with OpenCV
# Requires Python 3 and OpenCV 3

import cv2
import uuid


OUTPUT_IMG_SIZE = 200


def getFaces(image):

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


	for (x, y, w, h) in faces:
		img1 = image[y:y+h, x:x+w]

		#print ('Cropped image size is: ' + str(img1.shape))
		r = OUTPUT_IMG_SIZE / img1.shape[1]
		img2 = cv2.resize(img1, (OUTPUT_IMG_SIZE, OUTPUT_IMG_SIZE), interpolation = cv2.INTER_AREA)

		#generate a unique filename
		fname = "./" + str(uuid.uuid4()) + ".png"

		print("Saving face:", fname)
		cv2.imwrite(fname, img2)



# Load the sample image
getFaces(cv2.imread('data/abba1.jpg'))
getFaces(cv2.imread('data/abba2.jpg'))
getFaces(cv2.imread('data/abba3.jpg'))


# Wait for any key before exiting
cv2.waitKey(0)

