"""Image augmentations using torchvision transforms only."""
import torchvision.transforms as T
from typing import Tuple


def get_train_transforms(image_size: Tuple[int, int] = (512, 256)) -> T.Compose:
    """Training augmentations for pasture images."""
    w, h = image_size
    return T.Compose([
        T.RandomResizedCrop(
            size=(h, w),
            scale=(0.6, 1.0),
            ratio=(1.8, 2.2),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.BILINEAR),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: Tuple[int, int] = (512, 256)) -> T.Compose:
    """Validation/test transforms (clean resize)."""
    w, h = image_size
    return T.Compose([
        T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_tta_transforms(image_size: Tuple[int, int] = (512, 256)) -> list:
    """Test-time augmentation: return list of transforms for 5 views."""
    w, h = image_size
    base = [
        T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC),
    ]
    views = []

    # Center crop (standard view)
    views.append(T.Compose(base + [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    # Horizontal flip
    views.append(T.Compose(base + [T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    # Slight rotation variants
    for angle in [5, -5]:
        views.append(T.Compose(base + [T.RandomRotation(degrees=(angle, angle), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    # Zoom variant
    views.append(T.Compose([T.Resize((int(h * 1.1), int(w * 1.1)), interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop((h, w)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    return views


def get_contrastive_transforms(image_size: Tuple[int, int] = (512, 256)) -> T.Compose:
    """Stronger augmentations for SimCLR-style contrastive pre-training."""
    w, h = image_size
    return T.Compose([
        T.RandomResizedCrop(size=(h, w), scale=(0.5, 1.0), ratio=(1.8, 2.2), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.1),
        T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
