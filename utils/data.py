import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImagenetDataset(Dataset):
    """Imagenet dataset."""

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
                                str(self.image_frame.iloc[idx, 0]))
        image = Image.open(img_name)
        label = self.image_frame.iloc[idx, 1]

        if self.transform_type == 0:
            image = self.transform((image, 0, 0, 0))
        elif self.transform_type == 1:
            a, b = np.random.choice(self.num_variations, 2, replace=False)
            var1_name = os.path.join(self.root_dir,
                                     str(self.image_frame.iloc[idx, 2+a]))
            var1 = Image.open(var1_name)
            var2_name = os.path.join(self.root_dir,
                                     str(self.image_frame.iloc[idx, 2+b]))
            var2 = Image.open(var2_name)
            image = self.transform((var1, var2, 0, 1))
        elif self.transform_type == 2:
            a, b = np.random.choice(self.num_variations, 2, replace=False)
            var1_name = os.path.join(self.root_dir,
                                     str(self.image_frame.iloc[idx, 2+a]))
            var1 = Image.open(var1_name)
            var2_name = os.path.join(self.root_dir,
                                     str(self.image_frame.iloc[idx, 2+b]))
            var2 = Image.open(var2_name)
            image = self.transform((image, var1, var2, 2))
        elif self.transform_type == 3:
            a = np.random.choice(self.num_variations, 1, replace=False)
            a = a[0]
            var1_name = os.path.join(self.root_dir,
                                     str(self.image_frame.iloc[idx, 2+a]))
            var1 = Image.open(var1_name)
            image = self.transform((image, var1, 0, 3))
        elif self.transform_type == 5:
            a = np.random.choice(self.num_variations, 1, replace=False)
            a = a[0]
            var1_name = os.path.join(self.root_dir,
                                     str(self.image_frame.iloc[idx, 2+a]))
            var1 = Image.open(var1_name)
            image = self.transform(var1)
        else:
            image = self.transform(image)

        return image, label


class PetsDataset(Dataset):
    """Oxford-IIIT Pets dataset."""

    def __init__(self, root_dir, train, dogs, transform):
        """
        Arguments:
            root_dir (string): Directory with all the images of cute pets.
            train (bool): Whether to load the training or validation set.
            dogs (bool): Whether to load the cute dogs or everything.
            transform (callable, optional): Optional transform to be applied
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.dogs = dogs
        if self.train:
            if self.dogs:
                self.image_frame = pd.read_csv(
                    os.path.join(self.root_dir, 'train-dogs.csv'))
                self.root_dir = os.path.join(self.root_dir, 'train-dogs')
            else:
                self.image_frame = pd.read_csv(
                    os.path.join(self.root_dir, 'train.csv'))
                self.root_dir = os.path.join(self.root_dir, 'train')
        else:
            if self.dogs:
                self.image_frame = pd.read_csv(
                    os.path.join(self.root_dir, 'val-dogs.csv'))
                self.root_dir = os.path.join(self.root_dir, 'val-dogs')
            else:
                self.image_frame = pd.read_csv(
                    os.path.join(self.root_dir, 'val.csv'))
                self.root_dir = os.path.join(self.root_dir, 'val')

    def __len__(self):
        if self.train:
            if self.dogs:
                return 3490
            else:
                return 5170
        else:
            if self.dogs:
                return 1500
            else:
                return 2220

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                str(self.image_frame.iloc[idx, 0]))
        image = Image.open(img_name)
        # convert this image from jpeg to png
        image = image.convert('RGB')
        label = self.image_frame.iloc[idx, 1]

        image = self.transform(image)

        return image, label


class NegativeImagenetDataset(Dataset):
    # Purpose of class:
    # Obtaining negative image variations for all class labels
    # in Imagenette dataset

    def __init__(self, images_folder, batch_size, n_classes,
                 n_img_samples_per_class, class_remapping_file_path,
                 epochs, train_steps, steps_per_epoch, transform):
        # Expectations:
        # Image_folder exists and contains images for all class labels
        # (No assert, but this will crash the program later)
        # Batch size is less than or equal to the number of
        # available image files (Enforced by assert)
        super().__init__()
        self.max_classes = n_classes
        self.max_variations = n_img_samples_per_class
        self.epochs = epochs
        self.train_steps = train_steps
        self.steps_per_epoch = steps_per_epoch

        self.images_folder = images_folder
        self.batch_size = batch_size
        self.transform = transform

        self.size = self.batch_size * self.steps_per_epoch
        self.index_list = []

        assert len(
            [file_name for file_name in os.listdir(images_folder)]
        ) >= self.batch_size, \
            'Number of available files is less than the batch_size'
        self.len = self.batch_size

        self.num_variations = int(self.batch_size/self.max_classes)
        if self.num_variations == 0:
            self.num_variations = 1

        self.inverse_class_mappings = self.load_class_mappings(
            class_remapping_file_path)
        self.get_index_array()

    def __len__(self):
        return self.size

    def load_class_mappings(self, mapping_file_path):
        if mapping_file_path is None:
            return {
                class_id: class_id for class_id in np.arange(self.max_classes)
            }

        class_mappings = pd.read_pickle(mapping_file_path, compression='infer')
        return {
            r_class_id: class_id
            for class_id, r_class_id in class_mappings.items()
        }

    def get_index_array(self):
        class_arr = np.arange(self.max_classes)
        variation_arr = np.arange(self.max_variations)
        selected_variations = np.random.choice(variation_arr, size=self.size)
        if self.max_classes < self.batch_size:
            selected_classes = np.tile(np.arange(self.max_classes), int(np.ceil(self.size/self.max_classes)))
            selected_classes = selected_classes[:self.size]
        else:
            selected_class_list = []
            for i in range(self.steps_per_epoch):
                selected_classes = np.random.choice(class_arr, size=self.batch_size, replace=False)
                selected_class_list.append(selected_classes)
            selected_classes = np.array(selected_class_list).flatten()
        self.index_arr = np.stack((selected_classes, selected_variations), axis=1)

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

        class_index = int(self.inverse_class_mappings[self.index_arr[idx, 0]])
        variation_idx = int(self.index_arr[idx, 1])
        # class_index = int(self.class_indices[int(idx/self.num_variations)])
        # variation_idx = int(self.variation_indices[idx])
        img_name = os.path.join(self.images_folder,
                                f"{class_index}_{variation_idx}.png")
        image = Image.open(img_name)
        return self.transform(image), ''
