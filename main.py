'''

Identify clothes qnd their color in imqges
cqn do 1) with semantic segmentation models and 2) object detection and bounding boxes model

1) U-net for clothes segmentation with clothing-co-parsing


'''

import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
import segmentation_models_pytorch as smp
import argparse

DATAPATH = 'C:\\Users\\Elena\\PycharmProjects\\FashioProject\\clothing-co-parsing\\archive'

class DatasetExplorer:
    def __init__(self, name, datapath):
        self.name = name
        self.datapath = datapath

    def _get_metadata(self):
        metadata_df = pd.read_csv(os.path.join(self.datapath, 'metadata.csv'))
        metadata_df = metadata_df[['image_path', 'label_type', 'label_path']]
        metadata_df['image_path'] = metadata_df['image_path'].apply(lambda img_pth: os.path.join(self.datapath, img_pth))
        metadata_df['label_path'] = metadata_df['label_path'].apply(
            lambda lbl_pth: os.path.join(self.datapath, 'labels', lbl_pth))
        # Select data subset with pixel-level annotations (ignoring image-level annotations)
        metadata_df = metadata_df[metadata_df['label_type'] == 'pixel-level']
        # Shuffle DataFrame
        return metadata_df.sample(frac=1).reset_index(drop=True)

    def _get_class_names(self):
        class_dict = pd.read_csv(os.path.join(self.datapath, 'class_dict.csv'))
        # Get class names
        class_names = class_dict['class_name'].tolist()
        # Rewriting 'null' class to prevent a NaN value
        class_names[0] = 'null'
        # Get class RGB values
        class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()
        return class_names, class_rgb_values

    def _get_trainval_split(self):
        metadata = self._get_metadata()
        valid_df = metadata.sample(frac=0.05, random_state=42)
        train_df = metadata.drop(valid_df.index)
        return train_df, valid_df

    def _get_selected_classes(self):
        # Useful to shortlist specific classes in datasets with large number of classes
        select_classes = ['null', 'accessories', 'bag', 'belt', 'blazer', 'blouse', 'bodysuit',
                      'boots', 'bra', 'bracelet', 'cape', 'cardigan', 'clogs', 'coat', 'dress',
                      'earrings', 'flats', 'glasses', 'gloves', 'hair', 'hat', 'heels', 'hoodie',
                      'intimate', 'jacket', 'jeans', 'jumper', 'leggings', 'loafers', 'necklace',
                      'panties', 'pants', 'pumps', 'purse', 'ring', 'romper', 'sandals', 'scarf',
                      'shirt', 'shoes', 'shorts', 'skin', 'skirt', 'sneakers', 'socks', 'stockings',
                      'suit', 'sunglasses', 'sweater', 'sweatshirt', 'swimwear', 't-shirt', 'tie',
                      'tights', 'top', 'vest', 'wallet', 'watch', 'wedges']
        class_names, class_rgb_values = self._get_class_names()
        # Get RGB values of required classes
        select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
        select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]
        return select_classes, select_class_indices, select_class_rgb_values


def main(args):
    if args.explore_dataset:
        dse = DatasetExplorer('clothing-co-parsing', datapath=DATAPATH)
        class_names, class_rgb_values = dse._get_class_names()
        print('All dataset classes and their corresponding RGB values in labels:')
        print('\nClass Names: ', class_names)
        print('\nClass RGB values: ', class_rgb_values)
        classes, indices, rgb_values = dse._get_selected_classes()
        print('Selected classes and their corresponding RGB values in labels:')
        print('\nClass Names: ', classes, 'N: ', len(classes))
        print('\nClass RGB values: ', rgb_values, 'N: ', len(rgb_values))
        train_split, val_split = dse._get_trainval_split()
        print("Train split length: ", len(train_split), "Validation split length: ", len(val_split))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--explore_dataset', help='explore dataset only', action='store_true')
    parser.add_argument('--dataset', help='dataset name', required=True)
    parser.add_argument('--train', help='train model', action='store_true')
    parser.add_argument('--predict', help='run_inference', action='store_true')
    args = parser.parse_args()

    main(args)

