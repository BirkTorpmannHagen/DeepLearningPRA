# import os
# import torch
# from torch import Tensor
# from pathlib import Path
# from typing import List, Optional, Sequence, Union, Any, Callable
# from torchvision.datasets.folder import default_loader
# from pytorch_lightning import LightningDataModule
# from torch.utils.data import DataLoader, Dataset, random_split
# from torchvision import transforms
# from torchvision.datasets import CIFAR10
# from torchvision.datasets import ImageFolder
# import zipfile
# from vae.utils.general import check_dataset, colorstr
# from domain_datasets import *
# def wrap_dataset(dataset):
#     """
#     dumb utility function to make testing easier. standardizes datasets so that it works easier with the models and trainers
#     :param dataset:
#     :return:
#     """
#     class NewDataset(data.Dataset):
#         def __init__(self, dataset):
#             super().__init__()
#             self.dataset = dataset
#
#         def __getitem__(self, index):
#             image, label = self.dataset[index]
#             if image.shape[0]==1:
#                 image = image.repeat(3,1,1)
#             return image, label, 0
#
#         def __len__(self):
#             return len(self.dataset)
#
#     return NewDataset(dataset)
#
# class VAEDataset(LightningDataModule):
#     """
#     PyTorch Lightning data module
#
#     Args:
#         data_dir: root directory of your dataset.
#         train_batch_size: the batch size to use during training.
#         val_batch_size: the batch size to use during validation.
#         patch_size: the size of the crop to take from the original images.
#         num_workers: the number of parallel workers to create to load data
#             items (see PyTorch's Dataloader documentation for more details).
#         pin_memory: whether prepared items should be loaded into pinned memory
#             or not. This can improve performance on GPUs.
#     """
#
#     def __init__(
#         self,
#         train_set,
#         val_set,
#         train_batch_size: int = 8,
#         val_batch_size: int = 8,
#         patch_size: Union[int, Sequence[int]] = (256, 256),
#         num_workers: int = 0,
#         pin_memory: bool = False,
#         **kwargs,
#     ):
#         super().__init__()
#         self.train_batch_size = train_batch_size
#         self.val_batch_size = val_batch_size
#         if isinstance(patch_size, int):
#             self.patch_size = (patch_size, patch_size)
#         else:
#             self.patch_size = patch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.train_set = train_set
#         self.val_set = val_set
#     def setup(self,  stage: Optional[str] = None) -> None:
#         # train_transforms = transforms.Compose([
#         #                                       # transforms.CenterCrop(148), #2048 with, 4096 without...
#         #                                       transforms.Resize((32,32)),
#         #                                       transforms.ToTensor(),])
#         #
#         # val_transforms = transforms.Compose([
#         #                                     # transforms.CenterCrop(148),
#         #                                     transforms.Resize((32,32)),
#         #                                     transforms.ToTensor(),])
#
#         train_transforms = transforms.Compose([
#             # transforms.CenterCrop(148), #2048 with, 4096 without...
#             transforms.Resize((512, 512)),
#             transforms.ToTensor(), ])
#
#         val_transforms = transforms.Compose([
#             # transforms.CenterCrop(148),
#             transforms.Resize((512, 512)),
#             transforms.ToTensor(), ])
#
#         # self.train_dataset, self.val_dataset = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", fold="Kvasir", seed=0)
#         # self.train_dataset, self.val_dataset = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, train_transforms, val_transforms, context="dim", seed=0)
#         # self.train_dataset = wrap_dataset(CIFAR10(root='../../Datasets/cifar10', train=True, download=False, transform=train_transforms))
#         # self.val_dataset = wrap_dataset(CIFAR10(root='../../Datasets/cifar10', train=False, download=False, transform=train_transforms))
#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.train_batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#             pin_memory=self.pin_memory,
#         )
#
#     def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.val_batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#             pin_memory=self.pin_memory,
#         )
#
#     def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
#         return DataLoader(
#             self.val_dataset,
#             batch_size=8,
#             num_workers=self.num_workers,
#             shuffle=True,
#             pin_memory=self.pin_memory,
#         )
#
#
