import numpy as np
import torchvision


class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data
    example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                # with 0.5 probability
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.CenterCrop(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.variation_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    def __call__(self, vars):
        x, y, z, type = vars
        # Traditional SimCLR augmentations
        if type == 0:
            return self.train_transform(x), self.train_transform(x)
        # Image Variation only
        elif type == 1:
            return self.variation_transform(x), self.variation_transform(y)
        # Randomly choose between traditional SimCLR augmentations
        # and Image Variation
        elif type == 2:
            a = np.random.random()
            if a < 0.5:
                return self.train_transform(x), self.train_transform(x)
            else:
                return self.variation_transform(y), self.variation_transform(z)
        # One traditional SimCLR augmentation and one Image Variation
        elif type == 3:
            return self.train_transform(x), self.variation_transform(y)
