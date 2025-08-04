from testbeds.base import *

class Office31TestBed(BaseTestBed):
    def __init__(self, rep_model="vae", mode="severity", sampler="RandomSampler", batch_size=16):
        super().__init__( mode=mode, sampler=sampler, batch_size=batch_size)
        self.trans = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(), ])
        self.ind_train, self.ind_val, self.ind_test, self.ood_val, self.ood_test = build_office31_dataset("../../Datasets/office31", self.trans, self.trans)
        self.ood_contexts = self.ood_val.ood_contexts

        self.num_classes = num_classes = self.ind_train.num_classes

        self.classifier = ResNetClassifier.load_from_checkpoint(
            "classifier_logs/Office31/checkpoints/epoch=140-step=19881.ckpt", num_classes=num_classes,
            resnet_version=101).to("cuda").eval()
        self.glow = GlowPL.load_from_checkpoint("glow_logs/Office31Dataset/checkpoints/epoch=297-step=42018.ckpt", in_channel=3, n_flow=32, n_block=4, conv_lu=True, affine=True).cuda().eval()
        self.mode = mode

    def get_ood_dict(self):
        return {self.ood_contexts[0]: self.dl(self.ood_val), self.ood_contexts[1]: self.dl(self.ood_test)}

