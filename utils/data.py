import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

COUNT_IMAGENETTE_CLASSES = 10
COUNT_MAX_NEG_SAMPLE_VARIATIONS = 10


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
        self.transform_type = transform_type
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
            a = a[0]
            var1_name = os.path.join(self.root_dir,
                                     self.image_frame.iloc[idx, 2+a])
            var1 = Image.open(var1_name)
            image = self.transform((image, var1, 0, 3))
        else:
            image = self.transform(image)

        return image, label


class NegativeImagenetteDataset(Dataset):
    # Purpose of class:
    # Obtaining negative image variations for all class labels
    # in Imagenette dataset

    def __init__(self, images_folder, batch_size) -> None:
        # Expectations:
        # Image_folder exists and contains images for all class labels
        # (No assert, but this will crash the program later)
        # Batch size is less than or equal to the number of
        # available image files (Enforced by assert)
        super().__init__()
        self.max_classes = COUNT_IMAGENETTE_CLASSES
        self.max_variations = COUNT_MAX_NEG_SAMPLE_VARIATIONS

        self.images_folder = images_folder
        self.batch_size = batch_size
        assert len(
            [file_name for file_name in os.listdir(images_folder)]
        ) >= self.batch_size, \
            'Number of available files is less than the batch_size'
        self.len = self.batch_size

        self.num_variations = int(self.batch_size/self.max_classes)
        if self.num_variations == 0:
            self.num_variations = 1
        self.randomize_samples()

        print('max_classes = ', self.max_classes)
        print('max_variations = ', self.max_variations)
        print('images_folder = ', self.images_folder)
        print('batch_size = ', self.batch_size)
        print('num_variations = ', self.num_variations)
        print('len = ', self.len)
        print('class_indices = ', self.class_indices)
        print('variation_indices = ', self.variation_indices)

    def __len__(self):
        return self.batch_size

    def randomize_samples(self):
        # Meant to be called before every epoch
        # Randomizes samples that wil be picked
        self.class_indices = np.random.choice(
            np.arange(self.max_classes),
            size=min(self.max_classes, self.batch_size),
            replace=False)
        self.variation_indices = []
        for _ in self.class_indices:
            self.variation_indices = np.concatenate(
                (self.variation_indices, np.random.choice(
                    np.arange(self.max_variations),
                    size=self.num_variations,
                    replace=False)))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        class_index = self.class_indices[int(idx/len(self.class_indices))]
        variation_idx = self.variation_indices[idx]
        img_name = os.path.join(self.images_folder,
                                str(class_index) + '_' + str(variation_idx))
        print(img_name)
        image = Image.open(img_name)
        return image
