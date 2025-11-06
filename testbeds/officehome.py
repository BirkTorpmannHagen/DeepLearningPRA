from testbeds.base import *

class OfficeHomeTestBed(BaseTestBed):
    def __init__(self, model="resnet", mode="severity", sampler="RandomSampler", batch_size=16):
        super().__init__( model=model, mode=mode, sampler=sampler, batch_size=batch_size)
        self.trans = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(), ])
        self.ind_train, self.ind_val, self.ind_test, self.oods = build_officehome_dataset("../../Datasets/OfficeHome", self.trans, self.trans)

        self.num_classes = num_classes = self.ind_train.num_classes
        # self.ind, self.ind_test = random_split(self.ind, [0.5, 0.5])
        if self.model == "resnet":

            self.classifier = ResNetClassifier.load_from_checkpoint(
                "classifier_logs/resnet/OfficeHome/checkpoints/epoch=85-step=18748.ckpt", num_classes=num_classes,
                resnet_version=101).to("cuda").eval()
        elif self.model == "vit":
            self.classifier = ViTClassifier.load_from_checkpoint(
                "classifier_logs/vit/OfficeHome/checkpoints/epoch=11-step=2616.ckpt", num_classes=num_classes, batch_size=32, lr=1e-3
            ).to("cuda").eval()
        else:
            raise ValueError(f"model must be 'resnet' or 'vit', was: {self.model}")

        self.glow = GlowPL.load_from_checkpoint("glow_logs/OfficeHome/checkpoints/epoch=99-step=21800.ckpt", in_channel=3, n_flow=32, n_block=4, conv_lu=True, affine=True).cuda().eval()

        # self.rep_model = self.glow
        # self.vae = VanillaVAE(3, 512).to("cuda").eval()
        # self.rep_model = self.vae
        self.mode = mode

    def get_ood_dict(self):
        return {dataset_name: self.dl(dataset) for dataset_name, dataset in self.oods.items()}