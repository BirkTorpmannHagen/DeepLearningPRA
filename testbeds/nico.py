from testbeds.base import *

class NICOTestBed(BaseTestBed):

    def __init__(self, sample_size, mode="severity", sampler="RandomSampler", batch_size=16):
        super().__init__( mode=mode, sampler=sampler, batch_size=batch_size)
        self.trans = transforms.Compose([
                                                 transforms.Resize((512, 512)),
                                                 transforms.ToTensor(), ])
        self.num_classes = num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))

        self.num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
        self.contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
        self.ind_train, self.ind_val, self.ind_test, self.oods = build_nico_dataset( "../../Datasets/NICO++", self.trans, self.trans, ind_context="dim")
        self.contexts.remove("dim")
        # self.ind, self.ind_test = random_split(self.ind, [0.5, 0.5])

        self.classifier = ResNetClassifier.load_from_checkpoint(
           "classifier_logs/NICO/checkpoints/epoch=279-step=175000.ckpt", num_classes=num_classes,
            resnet_version=101).to("cuda").eval()
        self.glow = GlowPL.load_from_checkpoint("glow_logs/NICODataset/checkpoints/epoch=499-step=312500.ckpt",  in_channel=3, n_flow=32, n_block=4, conv_lu=True, affine=True).cuda().eval()
        self.mode=mode

    def get_ood_dict(self):
        return {dataset_name: self.dl(dataset) for dataset_name, dataset in self.oods.items()}
