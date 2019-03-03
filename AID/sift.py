import numpy as np
import cv2 as cv
image = cv.imread('table.jpg')
cv.imshow('Original', image)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
detector = cv.xfeatures2d.SIFT_create()
keypoints = detector.detect(gray)
cv.drawKeypoints(image, keypoints, image,
                 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('Star Keypoints', image)
cv.waitKey()
