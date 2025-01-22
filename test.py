import torch
import random
from PIL import Image
import torchvision.transforms.functional as F

def random_non_overlapping_crops_with_centers(image_tensor, crop_size):
    """
    Generate two random non-overlapping crops from an image tensor and report the center of each crop.
    
    Args:
        image_tensor (torch.Tensor): Input image as a tensor of shape (C, H, W).
        crop_size (tuple): Desired crop size as (height, width).
    
    Returns:
        (tuple): 
            - crop1_center: (x, y) coordinates of the center of the first crop.
            - crop2_center: (x, y) coordinates of the center of the second crop.
            - crop1_tensor: First cropped image as a tensor.
            - crop2_tensor: Second cropped image as a tensor.
    """
    img_height, img_width = image_tensor.shape[1], image_tensor.shape[2]
    crop_height, crop_width = crop_size
    
    if crop_height > img_height or crop_width > img_width:
        raise ValueError("Crop size must be smaller than the image dimensions.")
    
    def get_random_crop():
        """Generate random coordinates for a crop."""
        x = random.randint(0, img_width - crop_width)
        y = random.randint(0, img_height - crop_height)
        return x, y, x + crop_width, y + crop_height
    
    # Try generating non-overlapping crops
    for _ in range(100):  # Limit attempts to avoid infinite loops
        crop1 = get_random_crop()
        crop2 = get_random_crop()
        
        # Check for overlap
        overlap = (
            max(crop1[0], crop2[0]) < min(crop1[2], crop2[2]) and
            max(crop1[1], crop2[1]) < min(crop1[3], crop2[3])
        )
        if not overlap:
            break
    else:
        raise RuntimeError("Failed to generate non-overlapping crops after 100 attempts.")
    
    # Get the centers of the crops
    crop1_center = ((crop1[0] + crop1[2]) // 2, (crop1[1] + crop1[3]) // 2)
    crop2_center = ((crop2[0] + crop2[2]) // 2, (crop2[1] + crop2[3]) // 2)

    # Crop the image tensor
    crop1_tensor = image_tensor[:, crop1[1]:crop1[3], crop1[0]:crop1[2]]
    crop2_tensor = image_tensor[:, crop2[1]:crop2[3], crop2[0]:crop2[2]]

    return crop1_center, crop2_center, crop1_tensor, crop2_tensor

def manhattan_distance(center1, center2):
    """
    Calculate the Manhattan distance between two points.

    Args:
        center1 (tuple): Coordinates (x1, y1) of the first point.
        center2 (tuple): Coordinates (x2, y2) of the second point.

    Returns:
        float: The Manhattan distance.
    """
    return abs(center1[0] - center2[0]) + abs(center1[1] - center2[1])

def normalize_distance(distance, img_width, img_height):
    """
    Normalize the Manhattan distance based on image dimensions.

    Args:
        distance (float): The Manhattan distance to normalize.
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        float: The normalized Manhattan distance.
    """
    max_distance = img_width + img_height  # Max distance in the image (diagonal)
    return distance / max_distance

# Example usage
image_path = "/home/mahdi/Pictures/test.jpeg"  # Replace with the path to your image
output_dir = "/home/mahdi/Downloads/"
crop_size = (100, 100)  # Specify the crop size (height, width)

# Load the image and convert to tensor
image = Image.open(image_path).convert("RGB")
image_tensor = F.to_tensor(image)  # Convert image to tensor of shape (C, H, W)

# Generate the non-overlapping views and centers
crop1_center, crop2_center, view1_tensor, view2_tensor = random_non_overlapping_crops_with_centers(image_tensor, crop_size)

# Convert cropped tensors back to images
view1_image = F.to_pil_image(view1_tensor)
view2_image = F.to_pil_image(view2_tensor)

# Save the cropped views
view1_image.save(f"{output_dir}/view1.jpg")
view2_image.save(f"{output_dir}/view2.jpg")

# Calculate the Manhattan distance between the centers
distance = manhattan_distance(crop1_center, crop2_center)

# Normalize the Manhattan distance
normalized_distance = normalize_distance(distance, image_tensor.shape[2], image_tensor.shape[1])

# Print the results
print(f"Center of crop 1: {crop1_center}")
print(f"Center of crop 2: {crop2_center}")
print(f"Manhattan distance: {distance}")
print(f"Normalized Manhattan distance: {normalized_distance}")
