"""
Author: Tim Krashevsky
Date: 7/29/18

Contributors: OpenCV tutorials, Haar Caascade Tutorial
"""

import cv2 as cv
import numpy as np 
import os
import matplotlib.pyplot as plt
import copy

path = '/Users/timkrashevsky/anaconda3/lib/python3.6/site-packages/cv2/data/'

# Haar Caascade
face_cascade = cv.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(path + 'haarcascade_eye.xml')

img = cv.imread(os.getcwd() + '/Images' + '/kidredeye.jpg')
redeyeimg = cv.imread(os.getcwd() + '/Images' + '/kidredeye.jpg')
finalim = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

"""Currently, I'm using two individual loops/methods for differentiating between the right and left eye. 
There's probably a better way to do this, but all my other trials were only fixing one of the eyes"""

# extracing the right and left eyes

eyes_right = eye_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=6, minSize=(20, 20))
eyes_left = eye_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(100, 100))

# this loop splits the channels and extracts the red from the eye
for (x, y, w, h) in eyes_left:
	left_eye = img[y:y+h, x:x+w]
	# split the three channels 
	b = left_eye[:, :, 0]
	g = left_eye[:, :, 1]
	r = left_eye[:, :, 2]

	# sum up blue and green
	bg_sum = cv.add(b, g)

	# set condition for red
	mask = (r > 50) & (r > bg_sum)

	mask = mask.astype(np.uint8)*255

# loop for right eye
for (rx, ry, rw, rh) in eyes_right:
	right_eye = img[ry:ry+rh, rx:rx+rw]
	# split the three channels 
	b = right_eye[:, :, 0]
	g = right_eye[:, :, 1]
	r = right_eye[:, :, 2]

	# sum up blue and green
	bgr_sum = cv.add(b, g)

	# set condition for red
	mask2 = (r > 50) & (r > bgr_sum)

	mask2 = mask2.astype(np.uint8)*255


# fill for left eye and right eye are identical, besides variable names
def fill(mask):
	# threshold image
	mask_fill = mask.copy()

	# mask for flood fill
	x, y = mask_fill.shape[:2]
	temp_mask = np.zeros((y + 2, x + 2), np.uint8)

	# flood fill the image from the center
	cv.floodFill(mask_fill, temp_mask, (0,0), 255)

	# invert the flooded image
	mask_inv = cv.bitwise_not(mask_fill)
	return mask_inv | mask

def fillRight(mask):
	# threshold image
	mask_fill = mask.copy()

	# mask for flood fill
	x, y = mask_fill.shape[:2]
	temp_mask = np.zeros((y + 2, x + 2), np.uint8)

	# flood fill the image from the center
	cv.floodFill(mask_fill, temp_mask, (0,0), 255)

	# invert the flooded image
	mask_inv = cv.bitwise_not(mask_fill)
	return mask_inv | mask2

# flood the image for a grayscale mask
mask = fill(mask)
mask2 = fill(mask2)

# dilate the mask to ensure it captures rest of red-eye
mask = cv.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)
mask2 = cv.dilate(mask2, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)

"""Now the problem lies in replacing the masked red with the proper colors"""

# find average of blue and green to replace into new masked image
avg1 = bg_sum / 2
avg2 = bgr_sum / 2
mask = mask.astype(np.bool)[:, :, np.newaxis]
mask2 = mask2.astype(np.bool)[:, :, np.newaxis]
avg1 = avg1[:, :, np.newaxis]
avg2 = avg2[:, :, np.newaxis]

# put mean image onto the new eye image
eye1_copy = left_eye
eye2_copy = right_eye
np.copyto(eye1_copy, avg1.astype(np.uint8), where=mask)
np.copyto(eye2_copy, avg2.astype(np.uint8), where=mask2)

# combine all the channels and get the image
finalim[y:y+h, x:x+w, :] = eye1_copy
finalim[ry:ry+rh, rx:rx+rw, :] = eye2_copy

# final product!
np_hstack = np.hstack((redeyeimg, finalim))
cv.imshow('Red-Eye', np_hstack)
cv.waitKey(0)

