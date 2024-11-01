from njord.utils.dataloaders import LoadImagesAndLabels
from njord.utils.general import check_dataset


def create_dataset(path,
                   imgsz,
                   batch_size,
                   stride,
                   single_cls=False,
                   hyp=None,
                   augment=False,
                   cache=False,
                   pad=0.0,
                   rect=False,
                   rank=-1,
                   workers=8,
                   image_weights=False,
                   quad=False,
                   prefix='',
                   shuffle=False,
                   natively_trainable=False):
    dataset = NjordDataset(
        path,
        imgsz,
        batch_size,
        augment=augment,  # augmentation
        hyp=hyp,  # hyperparameters
        rect=rect,  # rectangular batches
        cache_images=cache,
        single_cls=single_cls,
        stride=int(stride),
        pad=pad,
        image_weights=image_weights,
        prefix=prefix,
        natively_trainable=natively_trainable)
    return dataset


def build_njord_datasets(img_size):


    ind = check_dataset("njord/folds/ind_fold.yaml")
    ood = check_dataset("njord/folds/ood_fold.yaml")

    train_set = create_dataset(ind["train"], img_size, 16, 32,natively_trainable=True)
    val_set =  create_dataset(ind["val"], img_size, 16, 32, natively_trainable=True)
    ood_set =  create_dataset(ood["val"], img_size, 16, 32, natively_trainable=True)
    return train_set, val_set, ood_set


class NjordDataset(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 prefix='',
                 natively_trainable=False):
        super().__init__(
                 path,
                 img_size,
                 batch_size,
                 augment,
                 hyp,
                 rect,
                 image_weights,
                 cache_images,
                 single_cls,stride,pad,prefix, natively_trainable)

    def __getitem__(self, item):
        x, targets, paths, shapes = super().__getitem__(item)
        return x, targets, paths, shapes

    def __len__(self):
        return min(super().__len__(), 10000)
