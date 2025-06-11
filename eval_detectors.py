# from yellowbrick.features import PCA
from panel.widgets.indicators import ptqdm

from testbeds import *


def compute_stats(train_features, train_losses, ind_val_features, ind_val_losses, ind_test_features, ind_test_losses, ood_features, ood_losses, fname, feature_names):
    dfs = convert_to_pandas_df(train_features, train_losses, ind_val_features, ind_val_losses, ind_test_features, ind_test_losses, ood_features, ood_losses, feature_names)
    for df, feature_name in zip(dfs, feature_names):
        df.to_csv(f"{fname}_{feature_name}.csv")


def collect_data(testbed_constructor, dataset_name, mode="noise"):
    bench = testbed_constructor("classifier", mode=mode)
    # features = [mahalanobis]
    features = [cross_entropy, grad_magnitude, energy, softmax, knn, typicality]

    # features = [knn]
    tsd = FeatureSD(bench.classifier,features)
    tsd.register_testbed(bench)
    compute_stats(*tsd.compute_pvals_and_loss(),
                  fname=f"final_data/{dataset_name}_{mode}", feature_names=[f.__name__ for f in features])

def collect_model_wise_data(testbed_constructor, dataset_name, mode="noise"):
    for model_name in ["deeplabv3plus", "unet", "segformer"]:
        bench = testbed_constructor("classifier", mode=mode, model_name=model_name)
        # features = [mahalanobis]
        features = [cross_entropy, grad_magnitude, energy, mahalanobis, softmax, knn, typicality]

        # features = [knn]
        tsd = FeatureSD(bench.classifier,features)
        tsd.register_testbed(bench)
        compute_stats(*tsd.compute_pvals_and_loss(),
                      fname=f"final_data/{dataset_name}_{mode}_{model_name}", feature_names=[f.__name__ for f in features])

    # bench = testbed_constructor("glow", mode=mode)
    # features = [typicality]
    # tsd = FeatureSD(bench.glow, features)
    # tsd.register_testbed(bench)
    # compute_stats(*tsd.compute_pvals_and_loss(),
    #               fname=f"single_data/{dataset_name}_{mode}", feature_names=[f.__name__ for f in features])



if __name__ == '__main__':
    from features import *
    torch.multiprocessing.set_start_method('spawn')

    # collect_data(PolypTestBed, "Polyp", mode="normal")
    # collect_data(PolypTestBed, "Polyp", mode="noise")
    # collect_data(PolypTestBed, "Polyp", mode="hue")
    # collect_data(PolypTestBed, "Polyp", mode="smear")
    # collect_data(PolypTestBed, "Polyp", mode="saturation")
    # collect_data(PolypTestBed, "Polyp", mode="brightness")
    # collect_data(PolypTestBed, "Polyp", mode="contrast")
    # collect_data(PolypTestBed, "Polyp", mode="multnoise")
    # collect_data(PolypTestBed, "Polyp", mode="saltpepper")
    # collect_data(PolypTestBed, "Polyp", mode="fgsm")


    # collect_data(CCTTestBed, "CCT", mode="normal")
    # collect_data(CCTTestBed, "CCT", mode="noise")
    # collect_data(CCTTestBed, "CCT", mode="hue")
    # collect_data(CCTTestBed, "CCT", mode="smear")
    # collect_data(CCTTestBed, "CCT", mode="saturation")
    # collect_data(CCTTestBed, "CCT", mode="brightness")
    # collect_data(CCTTestBed, "CCT", mode="contrast")
    # collect_data(CCTTestBed, "CCT", mode="multnoise")
    # collect_data(CCTTestBed, "CCT", mode="saltpepper")
    # collect_data(CCTTestBed, "CCT", mode="fgsm")

    collect_data(OfficeHomeTestBed, "OfficeHome", mode="normal")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="noise")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="hue")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="smear")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="saturation")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="brightness")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="contrast")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="multnoise")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="saltpepper")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="fgsm")

    collect_data(Office31TestBed, "Office31", mode="normal")
    # collect_data(Office31TestBed, "Office31", mode="noise")
    # collect_data(Office31TestBed, "Office31", mode="hue")
    # collect_data(Office31TestBed, "Office31", mode="smear")
    # collect_data(Office31TestBed, "Office31", mode="saturation")
    # collect_data(Office31TestBed, "Office31", mode="brightness")
    # collect_data(Office31TestBed, "Office31", mode="contrast")
    # collect_data(Office31TestBed, "Office31", mode="multnoise")
    # collect_data(Office31TestBed, "Office31", mode="saltpepper")
    # collect_data(Office31TestBed, "Office31", mode="fgsm")

    collect_data(NicoTestBed, "NICO", mode="normal")
    # collect_data(NicoTestBed, "NICO", mode="noise")
    # collect_data(NicoTestBed, "NICO", mode="hue")
    # collect_data(NicoTestBed, "NICO", mode="smear")
    # collect_data(NicoTestBed, "NICO", mode="saturation")
    # collect_data(NicoTestBed, "NICO", mode="brightness")
    # collect_data(NicoTestBed, "NICO", mode="contrast")
    # collect_data(NicoTestBed, "NICO", mode="multnoise")
    # collect_data(NicoTestBed, "NICO", mode="saltpepper")
    # collect_data(NicoTestBed, "NICO", mode="fgsm")

    # bench = NjordTestBed(10)
    # bench.split_datasets()