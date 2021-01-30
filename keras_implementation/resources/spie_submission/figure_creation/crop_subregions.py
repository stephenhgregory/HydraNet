"""
Used to create cropped subregions of result images and draw those cropped subregions as rectangles for figures in paper
"""

import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


image_dir = 'images'
new_image_dir = 'new_images'

# Blue crop coordinates
blue_x1 = 100
blue_x2 = 130
blue_y1 = 88
blue_y2 = 128

# Teal crop coordinates
yellow_x1 = 125
yellow_x2 = 170
yellow_y1 = 138
yellow_y2 = 178


if __name__ == "__main__":
    for image_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_name))

        image_name_no_ext, _ = os.path.splitext(image_name)
        
        # Save the blue crop
        crop_image_name = f'{os.path.join(new_image_dir, image_name_no_ext)}bluecrop.png'
        crop_image = image[blue_y1:blue_y2, blue_x1:blue_x2]
        cv2.imwrite(crop_image_name, crop_image)

        # Save the yellow crop
        crop_image_name = f'{os.path.join(new_image_dir, image_name_no_ext)}tealcrop.png'
        crop_image = image[yellow_y1:yellow_y2, yellow_x1:yellow_x2]
        cv2.imwrite(crop_image_name, crop_image)

        ## Draw rectangles ##
        rect_image_name = f'{os.path.join(new_image_dir, image_name_no_ext)}rect.png'
        # blue rectangle
        rect_image = cv2.rectangle(image, (blue_x1, blue_y1), (blue_x2, blue_y2), (255, 0, 0), 2)
        # yellow rectangle
        rect_image = cv2.rectangle(rect_image, (yellow_x1, yellow_y1), (yellow_x2, yellow_y2), (255, 255, 0), 2)
        # Save rectangle image
        cv2.imwrite(rect_image_name, rect_image)

