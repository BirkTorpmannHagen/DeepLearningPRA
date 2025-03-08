from segmentor.deeplab import SegmentationModel
from pytorch_lightning import Trainer
from datasets.polyps import build_polyp_dataset
from torch.utils.data import DataLoader
import warnings
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

warnings.filterwarnings('ignore')

#

def train_segmentor():
    model = SegmentationModel(transfer=False, batch_size=16)
    # model = SegmentationModel.load_from_checkpoint("segmentation_logs/lightning_logs/version_4/checkpoints/epoch=199-step=20000.ckpt", resnet_version=34)
    logger = TensorBoardLogger(save_dir="segmentation_logs")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",           # Metric to monitor
        mode="min",                   # Minimize the metric for best model (use "max" for metrics where higher is better)
        save_top_k=1,                 # Save only the best model
        dirpath="segmentation_logs/checkpoints",  # Directory to save the checkpoint
        filename="best"    # Filename for the checkpoint
    )
    trainer = Trainer(accelerator="gpu", max_epochs=300,logger=logger, callbacks=[checkpoint_callback])
    # trans = transforms.Compose([
    #                     transforms.Resize((512,512)),
    #                     transforms.ToTensor(), ])
    ind, val, test, _, _, _ = build_polyp_dataset("../../Datasets/Polyps", img_size=512)
    train_loader = DataLoader(ind, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val, batch_size=16, shuffle=True, num_workers=4)
    trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)

if __name__ == '__main__':
    train_segmentor()

