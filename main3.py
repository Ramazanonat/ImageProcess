from skimage.io import imread, imshow
from skimage.color import rgb2gray

lights = imread('images/lambs.jpeg')
lights_gray = rgb2gray(lights)

imshow(lights_gray);

template = lights_gray[79:132, 46:83]
imshow(template);

from skimage.feature import match_template

result = match_template(lights_gray, template)
imshow(result, cmap='viridis');

import numpy as np

x, y = np.unravel_index(np.argmax(result), result.shape)
print((x, y))

import matplotlib.pyplot as plt

imshow(lights_gray)
template_width, template_height = template.shape
rect = plt.Rectangle((y, x), template_height, template_width, color='y',
                     fc='none')
plt.gca().add_patch(rect);

from skimage.feature import peak_local_max

imshow(lights_gray)
template_width, template_height = template.shape
# set threshold initially at 0.65
for x, y in peak_local_max(result, threshold_abs=0.65):
    rect = plt.Rectangle((y, x), template_height, template_width, color='y',
                         fc='none')
    plt.gca().add_patch(rect);
