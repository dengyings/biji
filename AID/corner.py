import numpy as np
import cv2 as cv
image = cv.imread('box.png')
cv.imshow('Original', image)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
corner = cv.cornerHarris(gray, 7, 5, 0.04)
corner = cv.dilate(corner, None)
threshold = corner.max() * 0.01
corner_mask = corner > threshold
image[corner_mask] = [0, 0, 255]
cv.imshow('Corner', image)
cv.waitKey()
