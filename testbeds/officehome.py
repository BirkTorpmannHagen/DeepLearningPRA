from testbeds.base import *


class OfficeHomeTestBed(BaseTestBed):
    def __init__(self, sample_size, rep_model="vae", mode="severity"):
        super().__init__(sample_size)
        self.trans = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(), ])
        self.ind_train, self.ind_val, self.ind_test, self.oods = build_officehome_dataset("../../Datasets/OfficeHome", self.trans, self.trans)

        self.num_classes = num_classes = self.ind_train.num_classes
        # self.ind, self.ind_test = random_split(self.ind, [0.5, 0.5])

        self.classifier = ResNetClassifier.load_from_checkpoint(
            "train_logs/OfficeHome/checkpoints/epoch=197-step=43164.ckpt", num_classes=num_classes,
            resnet_version=101).to("cuda").eval()
        self.glow = Glow(3, 32, 4).cuda().eval()
        self.glow.load_state_dict(torch.load("glow_logs/OfficeHome_checkpoint/model_040001.pt"))
        # self.rep_model = self.glow
        # self.vae = VanillaVAE(3, 512).to("cuda").eval()
        # self.rep_model = self.vae
        self.mode = mode

    def get_ood_dict(self):
        return {dataset_name: self.dl(dataset) for dataset_name, dataset in self.oods.items()}