# from yellowbrick.features import PCA
from testbeds import *

def compute_stats_statistical(ind_pvalues, ood_pvalues_fold, ind_sample_losses, ood_sample_losses_fold, fname, feature_name):
    df = convert_stats_to_pandas_df(ind_pvalues, ood_pvalues_fold, ind_sample_losses, ood_sample_losses_fold, feature_name=feature_name)
    df.to_csv(fname)
def compute_stats(train_features, train_losses, ind_val_features, ood_features, ind_val_losses, ood_losses, fname, feature_names):
    dfs = convert_to_pandas_df(train_features, train_losses, ind_val_features, ood_features, ind_val_losses, ood_losses, feature_names)
    for df, feature_name in zip(dfs, feature_names):
        df.to_csv(f"{fname}_{feature_name}.csv")

def collect_gradient_data(sample_range, testbed_constructor, dataset_name, grad_fn, mode="normal", k=0):
    print(grad_fn)
    for sample_size in sample_range:
        if grad_fn==typicality or "Njord" in dataset_name:
            bench = testbed_constructor(sample_size, mode=mode, rep_model="vae")
            tsd = FeatureSD(bench.vae, grad_fn, k=k)
        else:
            bench = testbed_constructor(sample_size, "classifier", mode=mode)
            tsd = FeatureSD(bench.classifier, grad_fn,k=k)
        tsd.register_testbed(bench)
        if k!=0:
            name = f"new_data/{dataset_name}_{mode}_{grad_fn.__name__}_{k}NN_{sample_size}.csv"
        else:
            name = f"new_data/{dataset_name}_{mode}_{grad_fn.__name__}_{sample_size}.csv"
        compute_stats(*tsd.compute_pvals_and_loss(sample_size),
                      fname=name)

def collect_data(testbed_constructor, dataset_name, mode="noise"):
    bench = testbed_constructor("classifier", mode=mode)
    # features = [typicality]
    # features = [cross_entropy, knn, grad_magnitude, energy, softmax, typicality]
    features = [cross_entropy, knn, grad_magnitude, energy, softmax, typicality]
    # features = [softmax] #debug
    tsd = FeatureSD(bench.classifier,features)
    tsd.register_testbed(bench)
    compute_stats(*tsd.compute_pvals_and_loss(),
                  fname=f"single_data/{dataset_name}_{mode}", feature_names=[f.__name__ for f in features])

    # bench = testbed_constructor("glow", mode=mode)
    # features = [typicality]
    # tsd = FeatureSD(bench.glow, features)
    # tsd.register_testbed(bench)
    # compute_stats(*tsd.compute_pvals_and_loss(),
    #               fname=f"single_data/{dataset_name}_{mode}", feature_names=[f.__name__ for f in features])

def grad_data():
    pass

if __name__ == '__main__':
    from features import *
    torch.multiprocessing.set_start_method('spawn')

    # collect_data(PolypTestBed, "Polyp", mode="normal")
    # collect_data(PolypTestBed, "Polyp", mode="noise")
    # collect_data(PolypTestBed, "Polyp", mode="hue")
    collect_data(PolypTestBed, "Polyp", mode="smear")
    # collect_data(PolypTestBed, "Polyp", mode="saturation")
    # collect_data(PolypTestBed, "Polyp", mode="brightness")
    # collect_data(PolypTestBed, "Polyp", mode="contrast")
    # collect_data(PolypTestBed, "Polyp", mode="multnoise")
    # collect_data(PolypTestBed, "Polyp", mode="saltpepper")
    # collect_data(PolypTestBed, "Polyp", mode="fgsm")


    # collect_data(ECCVTestBed, "ECCV", mode="normal")
    # collect_data(ECCVTestBed, "ECCV", mode="noise")
    # collect_data(ECCVTestBed, "ECCV", mode="hue")
    # collect_data(ECCVTestBed, "ECCV", mode="smear")
    # collect_data(ECCVTestBed, "ECCV", mode="saturation")
    # collect_data(ECCVTestBed, "ECCV", mode="brightness")
    # collect_data(ECCVTestBed, "ECCV", mode="contrast")
    # collect_data(ECCVTestBed, "ECCV", mode="multnoise")
    # collect_data(ECCVTestBed, "ECCV", mode="saltpepper")
    # collect_data(ECCVTestBed, "ECCV", mode="fgsm")

    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="normal")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="noise")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="hue")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="smear")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="saturation")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="brightness")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="contrast")

    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="multnoise")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="saltpepper")
    # collect_data(OfficeHomeTestBed, "OfficeHome", mode="fgsm")

    # collect_data(Office31TestBed, "Office31", mode="normal")
    # collect_data(Office31TestBed, "Office31", mode="noise")
    # collect_data(Office31TestBed, "Office31", mode="hue")
    # collect_data(Office31TestBed, "Office31", mode="smear")
    # collect_data(Office31TestBed, "Office31", mode="saturation")
    # collect_data(Office31TestBed, "Office31", mode="brightness")
    # collect_data(Office31TestBed, "Office31", mode="contrast")

    # collect_data(Office31TestBed, "Office31", mode="multnoise")
    # collect_data(Office31TestBed, "Office31", mode="saltpepper")
    # collect_data(Office31TestBed, "Office31", mode="fgsm")

    # collect_data(NicoTestBed, "NICO", mode="normal")
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