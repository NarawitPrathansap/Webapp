# cut_image.py
from PIL import Image
import sys
import os

def cut_image(image_path, left_filename, right_filename):
    img = Image.open(image_path)
    width, height = img.size
    frac = 0.6

    # Crop 60% from the left of the image
    crop_left_width = int(width * frac)
    cropped_left = img.crop((0, 0, crop_left_width, height))
    cropped_left.save(left_filename)

    # Crop 60% from the right of the image and flip it
    crop_right_width = width - crop_left_width
    cropped_right = img.crop((crop_right_width, 0, width, height))
    flipped_right_side = cropped_right.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_right_side.save(right_filename)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python cut_image.py <image_path> <left_image_output_path> <right_image_output_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    left_image_output_path = sys.argv[2]
    right_image_output_path = sys.argv[3]

    cut_image(image_path, left_image_output_path, right_image_output_path)
