import os
import re

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import os
#
# path = "icons"
# dir_list = os.listdir(path)
# dir_list.sort(key=lambda f: int(re.sub('\D', '', f)))
# print(dir_list)

from skimage.io import imread, imshow
from skimage.color import rgb2gray

import cv2

method = cv2.TM_SQDIFF_NORMED

# Read the images from the file
small_image = cv2.imread('icons/small.png')
large_image = cv2.imread('icons/big.png')

result = cv2.matchTemplate(small_image, large_image, method)

# We want the minimum squared difference
mn, _, mnLoc, _ = cv2.minMaxLoc(result)

# Draw the rectangle:
# Extract the coordinates of our best match
MPx, MPy = mnLoc

# Step 2: Get the size of the template. This is the same size as the match.
trows, tcols = small_image.shape[:2]

# Step 3: Draw the rectangle on large_image
cv2.rectangle(large_image, (MPx, MPy), (MPx + tcols, MPy + trows), (0, 0, 255), 2)

# Display the original image with the rectangle around the match.
cv2.imshow('output', large_image)

# The image is only displayed if we call this
cv2.waitKey(0)

# img_rgb = cv.imread('mario.png')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# template = cv.imread('mario_coin.png', 0)
# w, h = template.shape[::-1]
# res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
# threshold = 0.8
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
# cv.imwrite('res.png', img_rgb)
