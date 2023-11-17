import numpy as np
import torch
import torchvision


class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
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
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
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

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class ImageVariations:
    """
    Picks a random variation of the image, converts it to a tensor,
    change the shape of the tensor to (C, H, W) and returns it.
    """

    def __init__(self):
        self.variation_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
        ])

    def __call__(self, vars):
        x, y = vars
        # x = x.astype(np.float32)
        # y = y.astype(np.float32)
        # x = torch.from_numpy(np.transpose(x, (2, 0, 1)))
        # y = torch.from_numpy(np.transpose(y, (2, 0, 1)))

        return self.variation_transform(x), self.variation_transform(y)
