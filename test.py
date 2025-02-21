import torch
import random
from PIL import Image
import torchvision.transforms.functional as F
from myrandomcrop import RandomResizedCrop

# Example usage
image_path = "/home/mahdi/Pictures/test.png"  # Replace with the path to your image
output_dir = "/home/mahdi/Downloads/"
crop = RandomResizedCrop(224)
# Load the image and convert to tensor
image = Image.open(image_path).convert("RGB")
# Convert image to tensor of shape (C, H, W)

# Generate the non-overlapping views and centers
view1_image, view2_image, view3_image, o = crop(image)

# Save the cropped views
view1_image.save(f"{output_dir}/view1.jpg")
view2_image.save(f"{output_dir}/view3.jpg")
view3_image.save(f"{output_dir}/view2.jpg")