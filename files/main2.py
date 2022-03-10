import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.feature import match_template, peak_local_max
from skimage import transform

plt.figure(figsize=(7, 7))

condo = imread('images/condo.jpg')
condo_gray = rgb2gray(condo)
imshow(condo)


def find_template(image, x, y, x_width, y_width):
    fig, ax = plt.subplots(1, 2, figsize=(14, 10))

    template = image[y:y + y_width, x:x + x_width]
    ax[0].imshow(template, cmap='gray')
    ax[0].set_title('Object Template')
    result = match_template(image, template)
    ax[1].imshow(result, cmap='viridis');
    ax[1].set_title('Cross-correlation image')

    coor_x, coor_y = np.unravel_index(np.argmax(result), result.shape)

    return coor_x, coor_y, template, result


x, y, template, result = find_template(condo_gray,
                                       x=2900, y=3350,
                                       x_width=250,
                                       y_width=300)
plt.tight_layout()

plt.figure(figsize=(6, 6))

imshow(condo)
template_width, template_height = template.shape
for x, y in peak_local_max(result, threshold_abs=0.5):
    rect = plt.Rectangle((y, x),
                         template_height,
                         template_width,
                         color='r',
                         fc='none')
    plt.gca().add_patch(rect);

area_of_interest = [(1359, 460),
                    (1931, 670),
                    (3830, 2800),
                    (1790, 2700)]

area_of_projection = [(1500, 1000),
                      (2500, 1000),
                      (2500, 4500),
                      (1500, 4500)]


def project_planes(image, src, dst):
    x_src = [val[0] for val in src] + [src[0][0]]
    y_src = [val[1] for val in src] + [src[0][1]]

    x_dst = [val[0] for val in dst] + [dst[0][0]]
    y_dst = [val[1] for val in dst] + [dst[0][1]]

    fig, ax = plt.subplots(1, 2, figsize=(13, 8))

    new_image = image.copy()
    projection = np.zeros_like(new_image)

    tform = transform.estimate_transform('projective',
                                         np.array(src),
                                         np.array(dst))
    transformed = transform.warp(image, tform.inverse)

    ax[0].imshow(new_image);
    ax[0].plot(x_src, y_src, 'r--')
    ax[0].set_title('Original Image')

    ax[1].imshow(transformed)
    ax[1].plot(x_dst, y_dst, 'r--')
    ax[1].set_title('Warped Image')
    plt.tight_layout()
    return transformed


transformed = project_planes(condo, area_of_interest, area_of_projection)

transformed_gray = rgb2gray(transformed)

x, y, template, result = find_template(transformed_gray,
                                       x=1970, y=4450,
                                       x_width=160,
                                       y_width=110)
plt.tight_layout()


def show_matches(image, template, result, x, y, threshold=1):
    plt.figure(figsize=(6, 6))

    imshow(image)
    template_width, template_height = template.shape
    for x, y in peak_local_max(result, threshold_abs=threshold):
        rect = plt.Rectangle((y, x),
                             template_height,
                             template_width,
                             color='r',
                             fc='none')
        plt.gca().add_patch(rect);


show_matches(transformed, template, result, x, y, threshold=1)

show_matches(transformed, template, result, x, y, threshold=0.8)

x, y, template, result = find_template(transformed_gray,
                                       x=1600, y=2640,
                                       x_width=100,
                                       y_width=110)
plt.tight_layout()

show_matches(transformed, template, result, x, y, 0.7)
