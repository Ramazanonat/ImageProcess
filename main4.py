import os
import cv2

path = "icons"
dir_list = sorted(filter(lambda x: os.path.isfile(os.path.join(path, x)), os.listdir(path)))

for idx, val in enumerate(dir_list):
    method = cv2.TM_SQDIFF_NORMED

    # Read the images from the file
    small_image = cv2.imread('icons/' + str(dir_list[idx]))
    large_image = cv2.imread('big.png')

    result = cv2.matchTemplate(small_image, large_image, method)

    # We want the minimum squared difference
    mn, _, mnLoc, _ = cv2.minMaxLoc(result)

    # Draw the rectangle:
    # Extract the coordinates of our best match
    MPx, MPy = mnLoc

    # Step 2: Get the size of the template. This is the same size as the match.
    trows, tcols = small_image.shape[:2]

    # Step 3: Draw the rectangle on large_image
    print("XT:" + str(MPx),
          "YT:" + str(MPy),
          "XB:" + str(MPx + tcols),
          "YB:" + str(MPy + trows)
          )
    cv2.rectangle(large_image, (MPx, MPy), (MPx + tcols, MPy + trows), (0, 0, 255), 2)

    # Display the original image with the rectangle around the match.
    # cv2.imshow('output', large_image)

    # The image is only displayed if we call this
    # cv2.waitKey(0)
