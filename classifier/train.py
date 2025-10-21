
from pytorch_lightning import Trainer
from datasets import *


from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings('ignore')

# torch and lightning imports
from torchvision import transforms
from torch.utils.data import DataLoader
from classifier.resnetclassifier import ResNetClassifier



def train_classifier(train_set, val_set, batch_size=16, load_from_checkpoint=None):
    num_classes =  train_set.num_classes
    model =  ResNetClassifier(num_classes, 101, transfer=False, batch_size=32, lr=1e-3).to("cuda")
    if load_from_checkpoint:
        model = ResNetClassifier.load_from_checkpoint(load_from_checkpoint, num_classes=num_classes, resnet_version=101)

    tb_logger = TensorBoardLogger(save_dir=f"classifier_logs/{type(train_set).__name__}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"classifier_logs/{type(train_set).__name__}/checkpoints",
        save_top_k=3,
        verbose=True,
        monitor="val_acc",
        mode="max"
    )

    trainer = Trainer(max_epochs=300, logger=tb_logger, accelerator="gpu",callbacks=checkpoint_callback)
    trainer.fit(model, train_dataloaders=DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4, persistent_workers=True),
                val_dataloaders=DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True))


if __name__ == '__main__':

    size = 512

    trans = transforms.Compose([transforms.Resize((size,size)),
                        transforms.ToTensor(),
                            transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomRotation(90),])
    val_trans = transforms.Compose([
                        transforms.Resize((size,size)),
                        transforms.ToTensor(), ])



    train_set, val_set, _, _ = build_officehome_dataset("../../Datasets/OfficeHome", train_transform=trans, val_transform=val_trans )
    train_classifier(train_set, val_set)

    train_set, val_set, _, _, _ = build_office31_dataset("../../Datasets/office31", train_transform=trans, val_transform=val_trans )
    train_classifier(train_set, val_set)

    train_set, val_set, _, _, _ = build_cct_dataset("../../Datasets/CCT", trans, val_trans)
    train_classifier(train_set, val_set)

    train_set, val_set, _, _ = build_nico_dataset("../../Datasets/NICO", train_transform=trans, val_transform=val_trans, ind_context="dim")
    train_classifier(train_set, val_set)


