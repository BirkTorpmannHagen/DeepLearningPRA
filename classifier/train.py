
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


# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.

<<<<<<< HEAD
def train_classifier(train_set, val_set, load_from_checkpoint=None):
    num_classes =  train_set.num_classes
    model =  ResNetClassifier(num_classes, 101, transfer=False, batch_size=32, lr=1e-4).to("cuda")
=======
def train_classifier(train_set, val_set, load_from_checkpoint=None, batch_size=32):
    num_classes =  train_set.num_classes
    model =  ResNetClassifier(num_classes, 34, transfer=True, batch_size=batch_size, lr=1e-3).to("cuda")
>>>>>>> ca53de764721a37a8d12d9ffc7a09053bd291ded

    if load_from_checkpoint:
        model = ResNetClassifier.load_from_checkpoint(load_from_checkpoint, num_classes=num_classes, resnet_version=101)
    # model = cifarrr
    tb_logger = TensorBoardLogger(save_dir=f"classifier_logs/{type(train_set).__name__}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"classifier_logs/{type(train_set).__name__}/checkpoints",
        save_top_k=3,
        verbose=True,
        monitor="val_acc",
        mode="max"
    )

    # ResNetClassifier.load_from_checkpoint("Imagenette_logs/checkpoints/epoch=82-step=24568.ckpt", resnet_version=101, nj
    trainer = Trainer(max_epochs=300, logger=tb_logger, accelerator="gpu",callbacks=checkpoint_callback)
<<<<<<< HEAD
    trainer.fit(model, train_dataloaders=DataLoader(train_set, shuffle=True, batch_size=16, num_workers=24),
                val_dataloaders=DataLoader(val_set, batch_size=16, shuffle=True, num_workers=24))
=======
    trainer.fit(model, train_dataloaders=DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4, persistent_workers=True),
                val_dataloaders=DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True))
>>>>>>> ca53de764721a37a8d12d9ffc7a09053bd291ded


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

    # train_set, val_set = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, val_trans, context="dim", seed=0)
    # train_set, val_set = build_imagenette_dataset("../../Datasets/imagenette2", train_trans=trans, val_trans=val_trans)
    train_set, val_set, test, ood_set = build_officehome_dataset("../../Datasets/OfficeHome", train_transform=trans, val_transform=val_trans )
    # train_classifier(train_set, val_set)

    # train_set, val_set, test_set, ood_val_set, ood_test_set = build_office31_dataset("../../Datasets/office31", train_transform=trans, val_transform=val_trans )ier(train_set, val_set, load_from_checkpoint="train_logs/OfficeHome/old_logs/epoch=153-step=33572.ckpt")
    train_classifier(train_set, val_set)
   #train_set, val_set, test_set, ood_val_set, ood_test_set = build_cct_dataset("../../Datasets/CCT", trans, val_trans)
   # train_classifier(train_set, val_set, load_from_checkpoint="classifier_logs/CCT/checkpoints/epoch=60-step=50813.ckpt")
    # train_set, val_set, ood_set = build_officehome_dataset("../../Datasets/OfficeHome", train_transform=trans, val_transform=val_trans)
    # train_set, test_set,val_set, ood_set = get_pneumonia_dataset("../../Datasets/Pneumonia", trans, val_trans)

    # CIAR10 and MNIST are already trained :D
