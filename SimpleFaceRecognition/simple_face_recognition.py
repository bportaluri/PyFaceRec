#!/usr/bin/python

# Simple face detection example with OpenCV
# Requires Python 3 and OpenCV 3

import cv2
import glob
import numpy as np



#---------------------------------------------------------------------------
def __get_images_and_labels(files):
	print("Loading faces for training")
	
	images = []
	labels = []

	c = 0
	for f in files:
		print(c, "-", f)
		#cv2.imshow("Adding faces to traning set...", cv2.imread(f))
		#cv2.waitKey(50)
		images.append(__prepare_image(f))
		labels.append(c)
		c = c+1
	
	cv2.destroyAllWindows()
	
	return images, labels

#---------------------------------------------------------------------------
# Load the image file, convert	to greyscale, normalize brightness and
# return the image
def __prepare_image(filename):
	img = cv2.imread(filename)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.equalizeHist(img)
	return img


def whoIs(fName):
	print(fName, end=' ')
	
	img = __prepare_image(fName)

	collector = cv2.face.StandardCollector_create()
	recognizer.predict_collect(img, collector)
	dist = collector.getMinDist()
	nbr_predicted = collector.getMinLabel()


	print(">>>", faceFiles[nbr_predicted], " (dist="+str(int(dist))+")")




#recognizer = cv2.face.createFisherFaceRecognizer()
#recognizer = cv2.face.createEigenFaceRecognizer()
recognizer = cv2.face.createLBPHFaceRecognizer()

faceFiles = glob.glob('facesdb/*/*.png')
images, labels = __get_images_and_labels(faceFiles)
recognizer.train(images, np.array(labels))


# Load the sample image
whoIs('data/Agnetha1.png')
whoIs('data/Agnetha2.png')
whoIs('data/AnniFrid1.png')
whoIs('data/AnniFrid2.png')
whoIs('data/Benny1.png')
whoIs('data/Benny2.png')
whoIs('data/Bjorn1.png')
whoIs('data/Bjorn2.png')


# Wait for any key before exiting
cv2.waitKey(0)

