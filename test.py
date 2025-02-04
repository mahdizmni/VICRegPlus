import torch
import random
from PIL import Image
import torchvision.transforms.functional as F
from myrandomcrop import RandomResizedCrop

# Example usage
image_path = "/home/mahdi/Pictures/test.jpeg"  # Replace with the path to your image
output_dir = "/home/mahdi/Downloads/"
crop_size = 100
m = 0.5

# Load the image and convert to tensor
image_tensor = Image.open(image_path).convert("RGB")
# image_tensor = F.to_tensor(image)  # Convert image to tensor of shape (C, H, W)
crop = RandomResizedCrop(crop_size)
# Generate the non-overlapping views and centers
crop1_img, crop2_img, c1, c2 = crop(image_tensor, m)


# Save the cropped views
crop1_img.save(f"{output_dir}/view1.jpg")
crop2_img.save(f"{output_dir}/view2.jpg")
print(c1)
print(c2)
