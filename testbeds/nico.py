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
        self.ind_train, self.ind_val, self.ind_test, self.oods = build_nico_dataset( "../../Datasets/NICO++", self.trans, self.trans, ind_context="dim")
        self.contexts.remove("dim")
        # self.ind, self.ind_test = random_split(self.ind, [0.5, 0.5])

        self.classifier = ResNetClassifier.load_from_checkpoint(
           "train_logs/NICO/checkpoints/epoch=279-step=175000.ckpt", num_classes=num_classes,
            resnet_version=101).to("cuda").eval()
        self.glow = Glow(3, 32, 4).cuda().eval()
        self.glow.load_state_dict(torch.load("glow_logs/NICODataset_checkpoint/model_040001.pt"))
        self.rep_model = self.glow
        self.mode=mode

    def get_ood_dict(self):
        return {dataset_name: self.dl(dataset) for dataset_name, dataset in self.oods.items()}
