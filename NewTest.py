import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

from auxiliary import draw_bounding_box, find_bounding_box, segment_pictures

image_path = "demo.jpg"
target_image_path = "target1.png"

img_grey = cv2.imread(image_path, 0)
target_img_grey = cv2.imread(target_image_path, 0)

Image.fromarray(img_grey)
Image.fromarray(target_img_grey)

main_pane = img_grey[:350, :]
color_threshold = 180
# main_pane = cv2.blur(main_pane, (3, 3))  # By blurring, we can remove some white pixels which may affecting the matching
main_pane = cv2.bilateralFilter(main_pane, 9, 75, 75)  # By blurring, we can remove some white pixels which may affecting the matching
main_pane[main_pane < color_threshold] = 0
main_pane[main_pane >= color_threshold] = 255
Image.fromarray(main_pane)
cv2.imshow('main', main_pane)
cv2.waitKey(0)

# cv2.imwrite('big4.jpg', main_pane)

icons_rect_coordinates = find_bounding_box(main_pane, (20, 20), (100, 100), sort=False)
icons = segment_pictures(main_pane, icons_rect_coordinates, (30, 30))
draw_bounding_box(main_pane, icons_rect_coordinates)

# Target
target_color_threshold = 40
target_pane = target_img_grey
target_pane[target_pane < target_color_threshold] = 0
target_pane[target_pane >= target_color_threshold] = 255

Image.fromarray(target_pane)
cv2.imshow('target', target_pane)
cv2.waitKey(0)

targets_rect_coordinates = find_bounding_box(target_pane, (5, 5), (100, 100))
targets = segment_pictures(target_pane, targets_rect_coordinates, (30, 30))

draw_bounding_box(target_pane, targets_rect_coordinates)


def calculate_max_matching(target, icon, d):
    largest_val = 0
    for degree in range(0, 360, d):
        tmp = ndimage.rotate(target, degree, reshape=False)
        res = cv2.matchTemplate(icon, tmp, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > largest_val:
            largest_val = max_val
    return largest_val


similarity_matrix = []
for target in targets:
    similarity_per_target = []
    for icon in icons:
        similarity_per_target.append(calculate_max_matching(target, icon, 6))
    similarity_matrix.append(similarity_per_target)

fig, ax = plt.subplots(1)
ax.imshow(main_pane)

# Calculate Mapping
target_candidates = [False for _ in range(len(targets))]
icon_candidates = [False for _ in range(len(icons))]

mapping = {}

# Sort the flatted similarity matrix in descending order, and assign the pair between target and icon if both of them
# havem't been assigned.
arr = np.array(similarity_matrix).flatten()
arg_sorted = np.argsort(-arr)

for e in arg_sorted:
    col = e // len(icons)
    row = e % len(icons)

    if target_candidates[col] == False and icon_candidates[row] == False:
        target_candidates[col], icon_candidates[row] = True, True
        mapping[col] = row

# Circling the most similar icon,
# blue circle: first target
# red circle: second target
# yellow circle: third target
# green circle: fourth target
color_map = {1: 'b', 2: 'r', 3: 'y', 4: 'g'}
for key in mapping:
    x, y, w, h = icons_rect_coordinates[mapping[key]]

    # x,y is the coordinate of top left hand corner
    # Bounding box is 70x70, so centre of circle = (x+70/2, y+70/2), i.e. (x+35, y+35)
    centre_x = x + (w // 2)
    centre_y = y + (h // 2)
    # Plot circle
    circle = plt.Circle((centre_x, centre_y), 20, color=color_map[key + 1], fill=False, linewidth=5)
    # Plot centre
    plt.plot([centre_x], [centre_y], marker='o', markersize=10, color="white")
    ax.add_patch(circle)

plt.show()
