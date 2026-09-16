import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


class POCTAugment:
    """
    Paired online augmentation for the RGB image and attention map.

    Geometric augmentation uses identical affine parameters for both
    branches. Photometric augmentation is applied only to the RGB image.
    """

    def __init__(self, always_apply=True):
        self.rotate_prob = 0.8
        self.noise_prob = 0.5
        self.contrast_prob = 0.5
        self.contrast_range = (0.8, 1.2)

        if always_apply:
            self.base_transform = T.Compose([
                T.Resize((256, 256)),
                T.Lambda(lambda image: image.convert("RGB")),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.base_transform = None

    @staticmethod
    def _add_gaussian_noise(img):
        """Add zero-mean Gaussian noise with standard deviation 0.05."""
        if isinstance(img, torch.Tensor):
            noise = torch.randn_like(img) * 0.05
            return torch.clamp(img + noise, 0.0, 1.0)

        img_array = np.asarray(img, dtype=np.float32)
        height, width = img_array.shape[:2]

        noise = np.random.normal(
            loc=0.0,
            scale=0.05,
            size=(height, width, 3),
        )

        noisy_img = np.clip(
            img_array + noise * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)

        return Image.fromarray(noisy_img)

    def _enhance_contrast(self, img):
        """Randomly adjust RGB-image contrast."""
        if isinstance(img, torch.Tensor):
            return img

        factor = random.uniform(*self.contrast_range)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    @staticmethod
    def _paired_random_affine(img, attention=None):
        """
        Apply the same affine parameters to the RGB image and attention map.

        RGB interpolation: bilinear.
        Attention interpolation: nearest neighbour.
        Fill value: zero.
        """
        angle, translate, scale, shear = T.RandomAffine.get_params(
            degrees=(-30.0, 30.0),
            translate=None,
            scale_ranges=(0.9, 1.1),
            shears=None,
            img_size=[img.width, img.height],
        )

        img = TF.affine(
            img,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

        if attention is not None:
            attention = TF.affine(
                attention,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        return img, attention

    def __call__(self, img, attention=None):
        operations = []

        if random.random() < self.rotate_prob:
            operations.append("affine")

        if random.random() < self.contrast_prob:
            operations.append("contrast")

        if random.random() < self.noise_prob:
            operations.append("noise")

        # Preserve the original implementation's random operation order.
        random.shuffle(operations)

        for operation in operations:
            if operation == "affine":
                img, attention = self._paired_random_affine(
                    img,
                    attention,
                )
            elif operation == "contrast":
                img = self._enhance_contrast(img)
            elif operation == "noise":
                img = self._add_gaussian_noise(img)

        if self.base_transform is not None:
            img = self.base_transform(img)

        if attention is None:
            return img

        return img, attention
