import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImagenetteDataset(Dataset):
    """Imagenette dataset."""

    def __init__(self, csv_file, root_dir, num_variations,
                 transform_type, transform):
        """
        Arguments:
            csv_file (string): Path to the csv file with image, labels and ids.
            root_dir (string): Directory with all the images.
            num_variations (int): Number of variations for each image.
            transform_type (int): Type of transformation to apply.
                0: Traditional SimCLR augmentations
                1: Image Variation only
                2: Randomly choose between traditional SimCLR augmentations
                   and Image Variation
                3: One traditional SimCLR augmentation and one Image Variation
                4: Test time augmentation (resize and center crop)
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
        image = Image.open(img_name)
        label = self.image_frame.iloc[idx, 1]

        if self.transform_type == 0:
            image = self.transform((image, 0, 0, 0))
        elif self.transform_type == 1:
            a, b = np.random.choice(self.num_variations, 2, replace=False)
            var1_name = os.path.join(self.root_dir,
                                     self.image_frame.iloc[idx, 2+a])
            var1 = Image.open(var1_name)
            var2_name = os.path.join(self.root_dir,
                                     self.image_frame.iloc[idx, 2+b])
            var2 = Image.open(var2_name)
            image = self.transform((var1, var2, 0, 1))
        elif self.transform_type == 2:
            a, b = np.random.choice(self.num_variations, 2, replace=False)
            var1_name = os.path.join(self.root_dir,
                                     self.image_frame.iloc[idx, 2+a])
            var1 = Image.open(var1_name)
            var2_name = os.path.join(self.root_dir,
                                     self.image_frame.iloc[idx, 2+b])
            var2 = Image.open(var2_name)
            image = self.transform((image, var1, var2, 2))
        elif self.transform_type == 3:
            a = np.random.choice(self.num_variations, 1, replace=False)
            var1_name = os.path.join(self.root_dir,
                                     self.image_frame.iloc[idx, 2+a])
            var1 = Image.open(var1_name)
            image = self.transform((image, var1, 0, 3))
        else:
            image = self.transform(image)

        return image, label
