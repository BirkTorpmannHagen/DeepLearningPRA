import sys
import os

import json
import numpy as np

import os
import json
import albumentations as alb
import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset
import numpy as np
from PIL import Image
from glob import glob
from torchvision.transforms import ToTensor
from torch.utils import data
import torchvision.transforms as transforms
from os import listdir
import torchvision
from os.path import join
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST
from torchvision.datasets import ImageFolder
from njord.utils.general import check_dataset
from njord.utils.dataloaders import LoadImagesAndLabels
from random import shuffle
import albumentations
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
import sklearn


class ECCV(Dataset):
    def __init__(self, path, train_transform, val_transform, fold="train"):
        self.data = []
        self.labels = []
        self.locations = []
        self.category_names = []
        self.category_labels = []
        self.data_dir = path
        self.extract_all_metadata()
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.fold = fold
        self.label_encoder = sklearn.preprocessing.LabelEncoder()
        if fold=="train":
            self.file_names = self.train_image_paths
            self.labels = self.train_labels
            self.locations = self.train_locations
        elif fold=="val":
            self.file_names = self.cis_val_image_paths
            self.labels = self.cis_val_labels
            self.locations = self.cis_val_locations
        elif fold=="test":
            self.file_names = self.cis_test_image_paths
            self.labels = self.cis_test_labels
            self.locations = self.cis_test_locations
        elif fold=="ood":

            self.file_names = list(self.trans_test_image_paths) + list(self.trans_val_image_paths)
            self.labels = list(self.trans_test_labels) + list(self.trans_val_labels)
            self.locations = list(self.trans_test_locations) + list(self.trans_val_locations)
        self.num_classes = len(np.unique(self.labels))
        self.label_encoder.fit(np.unique(self.labels))  # classes seen during training

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.fold=="train":
            img = self.train_transform(img)
        else:
            img = self.val_transform(img)
        label = self.label_encoder.transform([self.labels[index]])[0]
        return img, label

    def extract_all_metadata(self):
        def extract_metadata(path_to_json):
            with open(path_to_json) as json_file:
                data = json.load(json_file)

            # image_paths = np.array([os.path.join(self.data_dir,'small_images/eccv_18_all_images_sm',str(item['image_id'])+'.jpg') for item in data['annotations']])
            image_paths = np.array(
                [os.path.join(self.data_dir, 'standard_images/', str(item['image_id']) + '.jpg') for
                 item in data['annotations']])
            labels = np.array([int(item['category_id']) for item in data['annotations']])
            self.cat_dict = {int(item['id']): str(item['name']) for item in data['categories']}
            image_paths = image_paths[labels != 30]
            labels = labels[labels != 30]  # not present in the training set
            locations = np.array([int(item['location']) for item in data['images']])
            category_names = np.array([str(item['name']) for item in data['categories']])
            category_labels = np.array([int(item['id']) for item in data['categories']])
            return image_paths, np.squeeze(labels), locations, category_labels, category_names


        self.cis_test_image_paths, self.cis_test_labels, self.cis_test_locations, self.cis_test_category_labels, self.cis_test_category_names = extract_metadata(
            os.path.join(self.data_dir, 'annotations/cis_test_annotations.json'))

        self.cis_val_image_paths, self.cis_val_labels, self.cis_val_locations, self.cis_val_category_labels, self.cis_val_category_names = extract_metadata(
            os.path.join(self.data_dir, 'annotations/cis_val_annotations.json'))

        self.trans_test_image_paths, self.trans_test_labels, self.trans_test_locations, self.trans_test_category_labels, self.trans_test_category_names = extract_metadata(
            os.path.join(self.data_dir, 'annotations/trans_test_annotations.json'))

        self.trans_val_image_paths, self.trans_val_labels, self.trans_val_locations, self.trans_val_category_labels, self.trans_val_category_names = extract_metadata(
            os.path.join(self.data_dir, 'annotations/trans_val_annotations.json'))

        self.train_image_paths, self.train_labels, self.train_locations, self.train_category_labels, self.train_category_names = extract_metadata(
            os.path.join(self.data_dir, 'annotations/train_annotations.json'))

def build_ECCV(path, train_transform, val_transform):
    train = ECCV(path, train_transform, val_transform, fold="train")
    val = ECCV(path, train_transform, val_transform, fold="val")
    test = ECCV(path, train_transform, val_transform, fold="test")
    ood = ECCV(path, train_transform, val_transform, fold="ood")
    return train, val, test, ood




if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    trans = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
    dataset = ECCV("../../../Datasets/ECCV",trans, trans)
    for x, y in DataLoader(dataset, batch_size=1):
        plt.imshow(x[0].permute(1,2,0))
        plt.title(y)
        plt.show()
        input()

