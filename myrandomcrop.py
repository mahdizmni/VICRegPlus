import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=InterpolationMode.BICUBIC):
        super().__init__(size, scale, ratio, interpolation)

    def get_random_crop(self, img_width, img_height, crop_width, crop_height):
        """Generate random coordinates for a crop."""
        x = random.randint(0, img_width - crop_width)
        y = random.randint(0, img_height - crop_height)
        return x, y, x + crop_width, y + crop_height

    def random_non_overlapping_crops(self, image_tensor, crop_size):
        """
        Generate two non-overlapping crops and return the centers of each crop.
        """
        img_height, img_width = image_tensor.shape[1], image_tensor.shape[2]

        crop_height, crop_width = crop_size

        if crop_height > img_height or crop_width > img_width:
            raise ValueError("Crop size must be smaller than the image dimensions.")

        # Try generating non-overlapping crops
        for _ in range(100):  # Limit attempts to avoid infinite loops
            crop1 = self.get_random_crop(img_width, img_height, crop_width, crop_height)
            crop2 = self.get_random_crop(img_width, img_height, crop_width, crop_height)

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

        # A third interpolative crop        
        crop3_center = ((crop1_center[0] + crop2_center[0]) // 2, (crop1_center[1] + crop2_center[1]) // 2)

        x = crop3_center[0] - crop_width // 2
        y = crop3_center[1] - crop_height // 2
        crop3 = x, y, x + crop_width, y + crop_height

        # Crop the image tensor
        crop1_tensor = image_tensor[:, crop1[1]:crop1[3], crop1[0]:crop1[2]]
        crop2_tensor = image_tensor[:, crop2[1]:crop2[3], crop2[0]:crop2[2]]
        crop3_tensor = image_tensor[:, crop3[1]:crop3[3], crop3[0]:crop3[2]]

        return crop1_center, crop2_center, crop3_center, crop1_tensor, crop2_tensor, crop3_tensor

    def __call__(self, img):
        # Convert image to tensor
        img_tensor = F.to_tensor(img)

        # Apply RandomResizedCrop to one view
        crop1_center, crop2_center, crop3_center, crop1_tensor, crop2_tensor, crop3_tensor = self.random_non_overlapping_crops(img_tensor, self.size)
       # ? Apply the transformation to the crops / Is this even necessary?
        crop1_tensor = F.resize(crop1_tensor, self.size, interpolation=self.interpolation)
        crop2_tensor = F.resize(crop2_tensor, self.size, interpolation=self.interpolation)
        crop3_tensor = F.resize(crop3_tensor, self.size, interpolation=self.interpolation)

        # Convert cropped tensors back to images (optional)
        crop1_image = F.to_pil_image(crop1_tensor)
        crop2_image = F.to_pil_image(crop2_tensor)
        crop3_image = F.to_pil_image(crop3_tensor)

        # Return the two cropped images and their centers
        return crop1_image, crop2_image, crop3_image, crop1_center, crop2_center, crop3_center
