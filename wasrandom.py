import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=InterpolationMode.BICUBIC):
        super().__init__(size, scale, ratio, interpolation)

    def random_overlapping_crops(self, image_tensor, crop_size, m):
        """
        Generate two %m overlapping crops and return the centers of each crop.
        Note: m \in [0, 1)
        """
        img_height, img_width = image_tensor.shape[1], image_tensor.shape[2]
        crop_height, crop_width = crop_size

        if crop_height > img_height or crop_width > img_width:
            raise ValueError("Crop size must be smaller than the image dimensions.")

        non_overlap = True 
        # Do random crops when non-overlapping crops are not possible
        if (crop_height * 2 >= img_height or crop_width * 2 >= img_width) and m == 0:
            for _ in range(10):
                crop1 = self.get_random_crop(img_width, img_height, crop_width, crop_height)
                crop2 = self.get_random_crop(img_width, img_height, crop_width, crop_height)
                if crop1 != crop2:
                    non_overlap = False 
                    break
            
        if m == 0 and non_overlap:
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
            else: # Fall back
                # Pick two corners
                crop1 = (0, 0, crop_width, crop_height) 
                crop2 = (img_width - crop_width, img_height - crop_height, img_width, img_height)
#                raise RuntimeError("Failed to generate non-overlapping crops after 100 attempts.")

        elif m != 0: 
        # Try generating overlapping crops
            crop1 = self.get_random_crop(img_width, img_height, crop_width, crop_height)

            y = -1
            for _ in range(100):  # Limit attempts to avoid infinite loops
                x = random.randint(crop1[0], crop1[2])
                if x + crop_width > img_width or x == crop1[2]:
                    continue
                l = int(crop_width * crop_height * m // (crop1[2] - x))
                if l > crop_height:
                    continue
                
                y1 = crop1[3] - l
                y2 = crop1[1] - (crop_height - l) 

                # Checking for valid y
                if y1 + crop_height <= img_height and y2  >= 0:
                    p = random.randint(0, 1)
                    if p == 0:
                        y = y1 
                        break
                    else:
                        y = y2
                        break
                elif y1 + crop_height <= img_height: 
                    y = y1 
                    break
                elif y2 >= 0: 
                    y = y2 
                    break
                else:
                    continue
        
            if y == -1:
                raise RuntimeError("Failed to generate overlapping crops after 100 attempts.")

            crop2 = (x, y, x + crop_width, y + crop_height)

        # Get the centers of the crops
        crop1_center = ((crop1[0] + crop1[2]) // 2, (crop1[1] + crop1[3]) // 2)
        crop2_center = ((crop2[0] + crop2[2]) // 2, (crop2[1] + crop2[3]) // 2)

        # Crop the image tensor
        crop1_tensor = image_tensor[:, crop1[1]:crop1[3], crop1[0]:crop1[2]]
        crop2_tensor = image_tensor[:, crop2[1]:crop2[3], crop2[0]:crop2[2]]

        return crop1_center, crop2_center, crop1_tensor, crop2_tensor

    def __call__(self, img, m):
        # Convert image to tensor
        img_tensor = F.to_tensor(img)
        img_tensor = F.resize(img_tensor, 600, interpolation=self.interpolation)

        # Apply RandomResizedCrop to get two views
        crop1_center, crop2_center, crop1_tensor, crop2_tensor = self.random_overlapping_crops(img_tensor, self.size, m)

        # Apply the transformation to the crops
        crop1_tensor = F.resize(crop1_tensor, self.size, interpolation=self.interpolation)
        crop2_tensor = F.resize(crop2_tensor, self.size, interpolation=self.interpolation)

        # Convert cropped tensors back to images (optional)
        crop1_image = F.to_pil_image(crop1_tensor)
        crop2_image = F.to_pil_image(crop2_tensor)

        # Return the two cropped images and their centers
        return crop1_image, crop2_image, crop1_center, crop2_center
