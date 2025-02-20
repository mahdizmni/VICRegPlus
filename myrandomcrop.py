import torch
import numbers
import random
from torchvision.transforms import functional as F

class RandomResizedCrop(torch.nn.Module):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), 
                 interpolation=F.InterpolationMode.BICUBIC, antialias=None):
        super().__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.antialias = antialias

    def non_overlap_crop(self, img, scale, ratio):
        '''
        Softly try to generate non-overlapping views 
        '''
        for _ in range (100):
            t1, l1, ty1, lw1 = self.get_params(img, scale, ratio)
            t2, l2, ty2, lw2 = self.get_params(img, scale, ratio)
            if not (((l1 <= l2 <= lw1) or (l2 <= l1 <= lw2)) and ((t1 <= t2 <= ty1) or (t2 <= t1 <= ty2))):
                return (t1, l1, ty1, lw1), (t2, l2, ty2, lw2), 1
        crop1 = self.get_params(img, scale, ratio)
        crop2 = self.get_params(img, scale, ratio)
        return crop1, crop2, 0


    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop."""
        width, height = F.get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(random.uniform(*log_ratio))

            w = int(torch.round((target_area * aspect_ratio).sqrt()).item())
            h = int(torch.round((target_area / aspect_ratio).sqrt()).item())

            if 0 < w <= width and 0 < h <= height:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)
                return top, left, h, w

        # Fallback to central crop
        in_ratio = width / height
        if in_ratio < min(ratio):
            w = width
            h = int(w / min(ratio))
        elif in_ratio > max(ratio):
            h = height
            w = int(h * max(ratio))
        else:
            w = width
            h = height
        top = (height - h) // 2
        left = (width - w) // 2
        return top, left, h, w

    def inter_view(self, img, scale, ratio, c1, c2):
        """Get crop parameters centered around (center_x, center_y)."""
        width, height = F.get_image_size(img)
        area = height * width
        center_x = int((c1[0] + c2[0]) / 2)
        center_y = int((c1[1] + c2[1]) / 2)

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(random.uniform(*log_ratio))

            w = int(torch.round((target_area * aspect_ratio).sqrt()).item())
            h = int(torch.round((target_area / aspect_ratio).sqrt()).item())

            left = int(center_x - w / 2)
            top = int(center_y - h / 2)
            if 0 < left <= width and 0 < top <= height:
                return top, left, h, w

        # Fallback to central crop
        in_ratio = width / height
        if in_ratio < min(ratio):
            w = width
            h = int(w / min(ratio))
        elif in_ratio > max(ratio):
            h = height
            w = int(h * max(ratio))
        else:
            w = width
            h = height
        top = (height - h) // 2
        left = (width - w) // 2
        return top, left, h, w


    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        crop1, crop2, o = self.non_overlap_crop(img, self.scale, self.ratio)
        top, left, height, width = crop1
        c1 = (int((left + width) / 2), int((top + height) / 2))
        crop1 = F.resized_crop(
            img, top, left, height, width, self.size, 
            interpolation=self.interpolation, antialias=self.antialias
        ) 

        top, left, height, width = crop2
        c2 = (int((left + width) / 2), int((top + height) / 2))
        crop2 = F.resized_crop(
            img, top, left, height, width, self.size, 
            interpolation=self.interpolation, antialias=self.antialias
        ) 

        top, left, height, width = self.inter_view(img, self.scale, self.ratio, c1, c2)
        crop3 = F.resized_crop(
            img, top, left, height, width, self.size, 
            interpolation=self.interpolation, antialias=self.antialias
        ) 

        return crop1, crop2, crop3, o


    def __repr__(self):
        interpolate_str = self.interpolation.value if isinstance(self.interpolation, F.InterpolationMode) else self.interpolation
        return (f"{self.__class__.__name__}(size={self.size}, scale={self.scale}, ratio={self.ratio}, "
                f"interpolation={interpolate_str}, antialias={self.antialias})")
