import numpy as np
import cv2
from PIL import ImageGrab
from PIL import Image, ImageEnhance
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread
from skimage.color import rgb2yuv, rgb2hsv, rgb2gray, yuv2rgb, hsv2rgb
from scipy.signal import convolve2d
# left_x, top_y, right_x, bottom_y
coordinates = {(1308, 635, 1333, 660), (1331, 635, 1356, 660), (1355, 635, 1380, 660)}

time.sleep(5)

for idx, val in enumerate(coordinates):
    img = ImageGrab.grab(bbox=(val))  # x, y, w, h

    img = img.resize((50, 50))
    enhancer = ImageEnhance.Contrast(img)
    factor = 1.9  # decrease constrast
    img = enhancer.enhance(factor)

    img_np = np.array(img)
    img = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
    cv2.imwrite('icons/' + str(idx) + '.png', img_np),
    if cv2.waitKey(1) & 0Xff == ord('q'):
        break


# def convolver_rgb(image, kernel, iterations=1):
    # img_yuv = rgb2yuv(image)
    # img_yuv[:, :, 0] = multi_convolver(img_yuv[:, :, 0], kernel,
    #                                    iterations)
    # final_image = yuv2rgb(img_yuv)
    #
    # fig, ax = plt.subplots(1, 2, figsize=(17, 10))
    #
    # ax[0].imshow(image)
    # ax[0].set_title(f'Original', fontsize=20)
    #
    # ax[1].imshow(final_image);
    # ax[1].set_title(f'YUV Adjusted, Iterations = {iterations}',
    #                 fontsize=20)
    #
    # [axi.set_axis_off() for axi in ax.ravel()]
    #
    # fig.tight_layout()
    #
    # return final_image