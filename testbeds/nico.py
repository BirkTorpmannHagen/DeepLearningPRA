from testbeds.base import *

class NicoTestBed(BaseTestBed):

    def __init__(self, sample_size, rep_model="classifier", mode="severity"):
        super().__init__(sample_size)
        self.trans = transforms.Compose([
                                                 transforms.Resize((512, 512)),
                                                 transforms.ToTensor(), ])
        self.num_classes = num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))

        self.num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
        self.contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
        self.ind, self.ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, self.trans, self.trans, context="dim", seed=0)
        self.contexts.remove("dim")
        # self.ind, self.ind_test = random_split(self.ind, [0.5, 0.5])
        range1 = range(int(0.5*len(self.ind)))
        range2 = range(int(0.5*len(self.ind))+2, int(len(self.ind)))
        assert len(set(range1).intersection(range2))==0
        self.ind_test = Subset(self.ind, range1)
        self.ind=Subset(self.ind, range2)
        oods = [build_nico_dataset(1, "../../Datasets/NICO++", 0.2, self.trans, self.trans, context=context, seed=0)[1] for context in self.contexts]
        print(oods)
        self.ood = ConcatDataset(oods)
        self.classifier = ResNetClassifier.load_from_checkpoint(
           "NICODataset_logs/checkpoints/epoch=279-step=175000.ckpt", num_classes=num_classes,
            resnet_version=101).to("cuda").eval()
        self.glow = Glow(3, 32, 4).cuda().eval()
        self.glow.load_state_dict(torch.load("glow_logs/NICODataset_checkpoint/model_040001.pt"))
        self.rep_model = self.glow
        self.mode=mode
