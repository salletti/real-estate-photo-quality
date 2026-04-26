from torchvision import transforms

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms() -> transforms.Compose:
    """
    Pipeline de transformation des images pour le modèle.

    Étapes :
    1. Resize → uniformise la taille
    2. ToTensor → convertit en tensor (0–255 → 0–1)
    3. Normalize → adapte aux stats ImageNet
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])