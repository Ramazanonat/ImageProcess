import numpy as np
import cv2
from PIL import ImageGrab
# left_x, top_y, right_x, bottom_y
coordinates = {(1170, 530, 1270, 630), (1060, 530, 1150, 630), (950, 530, 1050, 630), (850, 530, 950, 630)}

for idx, val in enumerate(coordinates):
    img = ImageGrab.grab(bbox=(val))  # x, y, w, h
    img_np = np.array(img)
    cv2.imwrite('icons/' + str(idx) + '.png', img_np)
    if cv2.waitKey(1) & 0Xff == ord('q'):
        break
