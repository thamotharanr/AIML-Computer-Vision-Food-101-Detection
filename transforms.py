import torchvision.transforms as T
import torchvision.transforms.functional as F
import random

class ComposeWithTarget:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlipWithTarget:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            width = image.width
            boxes = target["boxes"]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())  # This will be applied last
    if train:
        transforms.insert(0, RandomHorizontalFlipWithTarget(0.5))
    return ComposeWithTarget(transforms)
