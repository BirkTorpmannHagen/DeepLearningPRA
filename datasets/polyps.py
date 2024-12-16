from os import listdir
from os.path import join

import albumentations as alb
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from torchvision import transforms as transforms
from torchvision.transforms import ToTensor


class KvasirSegmentationDataset(Dataset):
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
        # return 16 #debug
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


class EtisDataset(Dataset):
    """
        Dataset class that fetches Etis-LaribPolypDB images with the associated segmentation mask.
        Used for testing.
    """

    def __init__(self, path, trans):
        super(EtisDataset, self).__init__()
        self.path = path
        self.len = len(listdir(join(self.path, "Original")))
        self.transforms = trans
        self.tensor = ToTensor()

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        img_path = join(self.path, "Original/{}.jpg".format(i + 1))
        mask_path = join(self.path, "GroundTruth/p{}.jpg".format(i + 1))
        image = np.asarray(Image.open(img_path))
        mask = np.asarray(Image.open(mask_path))
        image, mask = self.transforms(image=image, mask=mask).values()

        return self.tensor(image), self.tensor(mask)[0].unsqueeze(0).int()


class CVC_ClinicDB(Dataset):
    def __init__(self, path, transforms):
        super(CVC_ClinicDB, self).__init__()

        self.path = path
        self.len = len(listdir(join(self.path, "Original")))
        indeces = range(self.len)
        self.train_indeces = indeces[:int(0.8*self.len)]
        self.val_indeces = indeces[int(0.8*self.len):]
        self.transforms = transforms
        self.common_transforms = transforms
        self.tensor = ToTensor()

    def __getitem__(self, i):
        img_path = join(self.path, "Original/{}.png".format(i+ 1))
        mask_path = join(self.path, "Ground Truth/{}.png".format(i + 1))
        image = np.asarray(Image.open(img_path))
        mask = np.asarray(Image.open(mask_path))
        image, mask = self.transforms(image=image, mask=mask).values()
        # mask = (mask>0.5).int()[0].unsqueeze(0)
        return self.tensor(image), self.tensor(mask)[0].unsqueeze(0).int()

    def __len__(self):
        # return 16 #debug
        #
        return self.len

class EndoCV2020(Dataset):
    def __init__(self, root_directory, tans):
        super(EndoCV2020, self).__init__()
        self.root = root_directory
        self.mask_fnames = listdir(join(self.root, "masksPerClass", "polyp"))
        self.mask_locs = [join(self.root, "masksPerClass", "polyp", i) for i in self.mask_fnames]
        self.img_locs = [join(self.root, "originalImages", i.replace("_polyp", "").replace(".tif", ".jpg")) for i in
                         self.mask_fnames]
        self.trans = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
        self.tensor = ToTensor()


    def __getitem__(self, i):
        image = Image.open(self.img_locs[i])
        mask = Image.open(self.mask_locs[i])
        image = self.trans(image)
        mask = self.trans(mask)
        return image, mask[0].unsqueeze(0).int()

    def __len__(self):
        # return 16 #debug
        return len(self.mask_fnames)


def build_polyp_dataset(root, img_size=512):
    train_trans = alb.Compose([alb.Resize(img_size, img_size), alb.HorizontalFlip(), alb.RandomRotate90(), alb.Transpose()])
    val_trans = alb.Compose([alb.Resize(img_size, img_size)])
    kvasir_root = join(root, "HyperKvasir")
    train_dataset = KvasirSegmentationDataset(kvasir_root, train_trans, val_trans, split="train")
    train_val = KvasirSegmentationDataset(kvasir_root, train_trans, val_trans, split="val")
    train_test = KvasirSegmentationDataset(kvasir_root, train_trans, val_trans, split="test")
    etis = EtisDataset(join(root, "ETIS-LaribPolypDB"), val_trans)
    cvc = CVC_ClinicDB(join(root, "CVC-ClinicDB"), val_trans)
    endo = EndoCV2020(join(root,"EndoCV2020"), val_trans)
    return train_dataset, train_val, train_test, etis, cvc, endo


if __name__ == '__main__':
    endocv2020 = EndoCV2020("../../../Datasets/Polyps/EndoCV2020", alb.Compose([alb.Resize(512, 512)]))
    loader = DataLoader(endocv2020, batch_size=1, shuffle=True)
    for i, (image, mask) in enumerate(loader):
        plt.imshow(image.squeeze(0).permute(1, 2, 0))
        plt.imshow(mask.squeeze(0).permute(1,2,0), alpha=0.5)
        plt.show()