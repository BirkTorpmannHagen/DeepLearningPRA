from os import listdir
from os.path import join

import albumentations as alb
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import ConcatDataset
from torchvision import transforms as transforms
from torchvision.transforms import ToTensor


class KvasirSegmentationDataset(data.Dataset):
    """
        Dataset class that fetches images with the associated segmentation mask.
    """
    def __init__(self, path, train_alb, val_alb, split="train"):
        super(KvasirSegmentationDataset, self).__init__()
        self.path = path
        self.fnames = listdir(join(self.path,"segmented-images", "images"))
        self.split = split
        self.train_transforms = train_alb
        self.val_transforms = val_alb
        train_size = int(len(self.fnames) * 0.8)
        val_size = (len(self.fnames) - train_size) // 2
        test_size = len(self.fnames) - train_size - val_size
        self.fnames_train = self.fnames[:train_size]
        self.fnames_val = self.fnames[train_size:train_size + val_size]
        self.fnames_test = self.fnames[train_size + val_size:]
        self.split_fnames = None  # iterable for selected split
        if self.split == "train":
            self.size = train_size
            self.split_fnames = self.fnames_train
        elif self.split == "val":
            self.size = val_size
            self.split_fnames = self.fnames_val
        elif self.split == "test":
            self.size = test_size
            self.split_fnames = self.fnames_test
        else:
            raise ValueError("Choices are train/val/test")
        self.tensor = ToTensor()


    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # img = Image.open(join(self.path, "segmented-images", "images/", self.split_fnames[index]))
        # mask = Image.open(join(self.path, "segmented-images", "masks/", self.split_fnames[index]))

        image = np.asarray(Image.open(join(self.path, "segmented-images", "images/", self.split_fnames[index])))
        mask =  np.asarray(Image.open(join(self.path, "segmented-images", "masks/", self.split_fnames[index])))
        if self.split=="train":
            image, mask = self.train_transforms(image=image, mask=mask).values()
        else:
            image, mask = self.val_transforms(image=image, mask=mask).values()
        image, mask = transforms.ToTensor()(Image.fromarray(image)), transforms.ToTensor()(Image.fromarray(mask))
        mask = torch.mean(mask,dim=0,keepdim=True).int()
        return image,mask


class EtisDataset(data.Dataset):
    """
        Dataset class that fetches Etis-LaribPolypDB images with the associated segmentation mask.
        Used for testing.
    """

    def __init__(self, path, val_alb, split="train"):
        super(EtisDataset, self).__init__()
        self.path = path
        self.len = len(listdir(join(self.path, "Original")))
        indeces = range(self.len)
        self.train_indeces = indeces[:int(0.8*self.len)]
        self.val_indeces = indeces[int(0.8*self.len):]
        self.transforms = val_alb
        self.split = split
        if self.split=="train":
            self.len=len(self.train_indeces)
        else:
            self.len=len(self.val_indeces)
        self.tensor = ToTensor()

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        if self.split=="train":
            index = self.train_indeces[i]
        else:
            index = self.val_indeces[i]


        img_path = join(self.path, "Original/{}.jpg".format(index + 1))
        mask_path = join(self.path, "GroundTruth/p{}.jpg".format(index + 1))
        image = np.asarray(Image.open(img_path))
        mask = np.asarray(Image.open(mask_path))
        image, mask = self.transforms(image=image, mask=mask).values()

        return self.tensor(image), self.tensor(mask)[0].unsqueeze(0).int()


class CVC_ClinicDB(data.Dataset):
    def __init__(self, path, transforms, split="train"):
        super(CVC_ClinicDB, self).__init__()
        self.path = path
        self.len = len(listdir(join(self.path, "Original")))
        indeces = range(self.len)
        self.train_indeces = indeces[:int(0.8*self.len)]
        self.val_indeces = indeces[int(0.8*self.len):]
        self.transforms = transforms
        self.split = split
        if self.split=="train":
            self.len=len(self.train_indeces)
        else:
            self.len=len(self.val_indeces)
        self.common_transforms = transforms
        self.tensor = ToTensor()

    def __getitem__(self, i):
        if self.split=="train":
            index = self.train_indeces[i]
        else:
            index = self.val_indeces[i]


        img_path = join(self.path, "Original/{}.png".format(index + 1))
        mask_path = join(self.path, "Ground Truth/{}.png".format(index + 1))
        image = np.asarray(Image.open(img_path))
        mask = np.asarray(Image.open(mask_path))
        image, mask = self.transforms(image=image, mask=mask).values()
        # mask = (mask>0.5).int()[0].unsqueeze(0)
        return self.tensor(image), self.tensor(mask)[0].unsqueeze(0).int()

    def __len__(self):
        return self.len


def build_polyp_dataset(root, ex=False, img_size=512):
    if ex:
        translist = [alb.Compose([
            i,
            alb.Resize(img_size, img_size)]) for i in [alb.HorizontalFlip(p=0), alb.HorizontalFlip(always_apply=True),
                                             alb.VerticalFlip(always_apply=True), alb.RandomRotate90(always_apply=True),
                                             ]]
    else:
        translist = [alb.Compose([
            alb.Resize(img_size, img_size)])]
    inds = []
    vals = []
    oods = []
    for trans in translist:
        cvc_train_set = CVC_ClinicDB(join(root, "CVC-ClinicDB"),trans, split="train")
        cvc_val_set = CVC_ClinicDB(join(root, "CVC-ClinicDB"),trans, split="val")
        kvasir_train_set = KvasirSegmentationDataset(join(root, "HyperKvasir"), train_alb=trans, val_alb=trans)
        kvasir_val_set = KvasirSegmentationDataset(join(root, "HyperKvasir"), train_alb=trans, val_alb=trans, split="val")
        etis_train_set = EtisDataset(join(root, "ETIS-LaribPolypDB"), trans, split="train")
        etis_val_set = EtisDataset(join(root, "ETIS-LaribPolypDB"), trans, split="val")

        inds.append(kvasir_train_set)
        inds.append(cvc_train_set)

        vals.append(kvasir_val_set)
        vals.append(cvc_val_set)

        oods.append(etis_train_set)
        oods.append(etis_val_set)

    ind = ConcatDataset(inds)
    ind_val = ConcatDataset(vals)
    ood = ConcatDataset(oods)
    return ind, ind_val, ood
