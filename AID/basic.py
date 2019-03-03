import numpy as np
import cv2 as cv
image = cv.imread('forest.jpg')
print(image.shape)
print(image)
cv.imshow('Original', image)
h, w = image.shape[:2]
l, t = int(w / 4), int(h / 4)
r, b = int(w * 3 / 4), int(h * 3 / 4)
cropped = image[t:b, l:r]
cv.imshow('Cropped', cropped)
blue = np.zeros_like(cropped)
green = np.zeros_like(cropped)
red = np.zeros_like(cropped)
blue[..., 0] = cropped[..., 0]
cv.imshow('Blue', blue)
green[..., 1] = cropped[..., 1]
cv.imshow('Green', green)
red[..., 2] = cropped[..., 2]
cv.imshow('Red', red)
scaled = cv.resize(cropped, (w, h), interpolation=cv.INTER_LINEAR)
cv.imshow('Scaled', scaled)
deformed = cv.resize(cropped, None, fx=2, fy=0.5,
                     interpolation=cv.INTER_LINEAR)
cv.imshow('Deformed', deformed)
cv.waitKey()
