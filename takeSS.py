import numpy as np
import cv2
from PIL import ImageGrab

coordinates = {(900, 200, 1000, 400), (910, 200, 1010, 400), (920, 200, 1020, 400), (930, 200, 1030, 400)}

for idx, val in enumerate(coordinates):
    img = ImageGrab.grab(bbox=(val))  # x, y, w, h
    img_np = np.array(img)
    cv2.imwrite('icons/' + str(idx) + '.png', img_np)
    if cv2.waitKey(1) & 0Xff == ord('q'):
        break
