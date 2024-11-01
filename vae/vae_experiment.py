import os
import math

import matplotlib.pyplot as plt
import torch
from torch import optim
from vae.models.base import BaseVAE
from vae.models.types_ import *
from vae.vae_utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, labels = batch[0].to(self.curr_device), batch[1].to(self.curr_device)
        self.curr_device = real_img.device

        results = self.forward(real_img)
        #Annealing for better training
        # M_N = 0.003*math.exp(-0.01*self.current_epoch)*math.sin(2*math.pi * self.current_epoch/ 100-math.pi/2) + 0.005 + 0.0005
        M_N = min(0.0005 + 0.00001 * self.current_epoch, 0.010)
        train_loss = self.model.loss_function(*results,
                                              M_N = M_N, #al_img.shape[0]/ self.num_train_imgs,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        self.log("m_n",M_N)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch[0].to(self.curr_device), batch[1].to(self.curr_device)
        self.curr_device = real_img.device
        results = self.forward(real_img)
        val_loss = self.model.loss_function(*results,
                                            M_N = 0.0005, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        batch = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input, test_label = batch[0], batch[1]
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label, _ = batch
        recons = self.model.generate(test_input)
        plt.imshow(recons[0].cpu().T)
        plt.show()
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir ,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(8,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir ,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):


        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, 2)
        return {"optimizer": optimizer, "scheduler":scheduler}
