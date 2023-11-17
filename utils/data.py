import os

import numpy as np
import pandas as pd
import torch
from skimage import io
from PIL import Image
from torch.utils.data import Dataset

from simclr.modules.transformations.simclr import ImageVariations


class ImagenetteDataset(Dataset):
    """Imagenette dataset."""

    def __init__(self, csv_file, root_dir, num_variations=0, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with image, labels and ids.
            root_dir (string): Directory with all the images.
            num_variations (int): Number of variations for each image.
            transform (callable, optional): Optional transform to be applied
        """
        self.image_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.num_variations = num_variations
        self.transform = transform

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.image_frame.iloc[idx, 0])
        image = io.imread(img_name)
        image = Image.fromarray(image)
        label = self.image_frame.iloc[idx, 1]

        if self.transform is not None:
            if self.num_variations > 0:
                a, b = np.random.choice(self.num_variations, 2, replace=False)
                var1_name = os.path.join(self.root_dir,
                                         self.image_frame.iloc[idx, 2+a])
                var1 = io.imread(var1_name)
                var2_name = os.path.join(self.root_dir,
                                         self.image_frame.iloc[idx, 2+b])
                var2 = io.imread(var2_name)
                image = self.transform((var1, var2))
            else:
                harsh = image
                image = self.transform(image)
                print(harsh.size, image.shape)

        return image, label
