from testbeds.base import *

class Office31TestBed(BaseTestBed):
    def __init__(self, sample_size, rep_model="vae", mode="severity"):
        super().__init__(sample_size)
        self.trans = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(), ])
        self.ind, self.ind_val, self.ood_val, self.ood_test = build_office31_dataset("../../Datasets/office31", self.trans, self.trans)

        self.num_classes = num_classes = self.ind.num_classes
        self.contexts = len(self.ind.contexts)
        # self.ind, self.ind_test = random_split(self.ind, [0.5, 0.5])
        range1 = range(int(0.5 * len(self.ind)))
        range2 = range(int(0.5 * len(self.ind)) + 2, int(len(self.ind)))
        assert len(set(range1).intersection(range2)) == 0
        self.ind_test = Subset(self.ind, range1)
        self.ind = Subset(self.ind, range2)

        self.classifier = ResNetClassifier.load_from_checkpoint(
            "Office31Dataset_logs/checkpoints/epoch=95-step=13536.ckpt", num_classes=num_classes,
            resnet_version=101).to("cuda").eval()
        self.glow = Glow(3, 32, 4).cuda().eval()
        self.glow.load_state_dict(torch.load("glow_logs/Office31Dataset_checkpoint/model_040001.pt"))
        # self.rep_model = self.glow
        # self.vae = VanillaVAE(3, 512).to("cuda").eval()
        # self.rep_model = self.vae
        self.mode = mode


