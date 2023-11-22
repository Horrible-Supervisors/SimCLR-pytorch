import numpy as np
import pandas as pd


class DataManipulator():
    """
    Data Manipulator class for choosing number of classes
    and number of images per class
    """

    def __init__(self, train_csv, val_csv, num_classes,
                 num_images_per_class_train, num_images_per_class_val):
        """
        train_csv: csv file containing image names, labels and variations
        val_csv: csv file containing image names and labels for validation
        num_classes: number of classes to be chosen
        num_images_per_class_train: number of images per class for training
        num_images_per_class_val: number of images per class for validation
        """
        self.train_frame = pd.read_csv(train_csv)
        self.val_frame = pd.read_csv(val_csv)
        self.num_classes = num_classes
        self.num_images_per_class_train = num_images_per_class_train
        self.num_images_per_class_val = num_images_per_class_val

    def create_csv(self, train_csv, val_csv, seed):
        """
        train_csv: csv file to be created
        val_csv: csv file to be created
        seed: seed for random sampling
        """
        # Randomly sample the given number of classes and images per class
        # from the original csv file
        # Save the new csv file
        np.random.seed(seed)
        classes = np.random.choice(self.train_frame["label"].unique(),
                                   self.num_classes,
                                   replace=False)

        df_train = pd.DataFrame()
        df_val = pd.DataFrame()
        for cls in classes:
            filtered_train = self.train_frame[self.train_frame["label"] == cls]
            sampled_train = filtered_train.sample(
                self.num_images_per_class_train)
            df_train = pd.concat([df_train, sampled_train], axis=0)

            filtered_val = self.val_frame[self.val_frame["label"] == cls]
            sampled_val = filtered_val.sample(self.num_images_per_class_val)
            df_val = pd.concat([df_val, sampled_val], axis=0)

        df_train = df_train.sort_values(
            by="image", key=lambda x: x.str.split(".").str[0].astype(int))
        df_val = df_val.sort_values(
            by="image", key=lambda x: x.str.split(".").str[0].astype(int))

        df_train.to_csv(train_csv, index=False)
        df_val.to_csv(val_csv, index=False)
