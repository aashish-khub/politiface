'''
$ python facial_landmarks.py
    --p shape_predictor_68_face_landmarks.dat
    --i unlabeled/file.jpg
    --o labeled/fileName.jpg
    
The p is optional, so we can just run:
$ python facial_landmarks.py --i unlabeled/file.jpg --o labeled/fileName.jpg   

Most of this code is detailed here:
    https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

'''

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-o", "--output", required=True,
    help="output path and file name")
args = vars(ap.parse_args())
#print(args)
if args["shape_predictor"] != None:
    trainingSetFilePath = args["shape_predictor"]
else:
    #set here the default training set path!
    trainingSetFilePath = "shape_predictor_68_face_landmarks.dat"

inputFilePath = args["image"]
outputFilePath = args["output"]
print()
print(f"Received file {inputFilePath}, loading up...")

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(trainingSetFilePath)

# load the input image, ~~resize it,~~ and convert it to grayscale
image = cv2.imread(args["image"])
#image = imutils.resize(image, width=500) #We don't want resizing!
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the face number
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)




print("LABELING COMPLETE!")
# show the output image with the face detections + facial landmarks
cv2.imwrite(outputFilePath,image)
print(f"Saved to file at path {outputFilePath}.\n")

#cv2.imshow("Output", image)    #This really really causes bugs for me...
#cv2.waitKey(10000)             #...so I abandoned it  -AK

