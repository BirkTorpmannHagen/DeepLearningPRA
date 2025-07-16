# from yellowbrick.features import PCA

from testbeds import *
from utils import load_all, BATCH_SIZES


def compute_stats(train_features, train_losses, ind_val_features, ind_val_losses, ind_test_features, ind_test_losses, ood_features, ood_losses, fname, feature_names):
    dfs = convert_to_pandas_df(train_features, train_losses, ind_val_features, ind_val_losses, ind_test_features, ind_test_losses, ood_features, ood_losses, feature_names)
    for df, feature_name in zip(dfs, feature_names):
        df.to_csv(f"{fname}_{feature_name}.csv")


def collect_data(testbed_constructor, dataset_name, mode="noise"):
    bench = testbed_constructor("classifier", mode=mode)
    # features = [mahalanobis]
    features = [cross_entropy, grad_magnitude, energy,knn, typicality, softmax]

    # features = [knn]
    tsd = FeatureSD(bench.classifier,features)
    tsd.register_testbed(bench)
    compute_stats(*tsd.compute_pvals_and_loss(),
                  fname=f"final_data/{dataset_name}_{mode}", feature_names=[f.__name__ for f in features])


def collect_debiased_data(testbed_constructor, dataset_name, mode="noise", sampler="RandomSampler", k=5, batch_size=8):
    bench = testbed_constructor("classifier", mode=mode, sampler=sampler, batch_size=batch_size)
    features = [cross_entropy, grad_magnitude, energy, typicality, softmax]
    # features = [typicality]
    # features = [rabanser_ks]
    tsd = BatchedFeatureSD(bench.classifier,features,k=k)
    tsd.register_testbed(bench)
    compute_stats(*tsd.compute_pvals_and_loss(),
                  fname=f"debiased_data/{dataset_name}_{mode}_{sampler}_{batch_size}_k={k}", feature_names=[f.__name__ for f in features])

def collect_rabanser_data(testbed_constructor, dataset_name, mode="noise", sampler="RandomSampler", k=5, batch_size=8):
    fname = f"{dataset_name}_{mode}_{sampler}_{batch_size}_k={k}_rabanser.csv"
    if fname in os.listdir("debiased_data"):
        print(f"{fname} already exists, skipping...")
        return
    print(f"Collecting data for {dataset_name} in {mode} mode with {sampler} sampler and batch size {batch_size} and k={k}")
    bench = testbed_constructor("classifier", mode=mode, sampler=sampler, batch_size=batch_size)
    tsd = RabanserSD(bench.classifier,k=k)
    tsd.register_testbed(bench)
    compute_stats(*tsd.compute_pvals_and_loss(),
                  fname=f"debiased_data/{dataset_name}_{mode}_{sampler}_{batch_size}_k={k}", feature_names=["rabanser"])


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


def collect_bias_data(k):
    # collect_data(PolypTestBed, "Polyp", mode="normal")
    for batch_size in [8,16]:
        # for sampler in ["RandomSampler","ClusterSampler",  "ClassOrderSampler"]:
        for sampler in ["RandomSampler", "ClusterSampler", "SequentialSampler"]:
            collect_debiased_data(PolypTestBed, "Polyp", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
            # collect_debiased_data(CCTTestBed, "CCT", mode="normal",k=k, sampler=sampler, batch_size=batch_size)
            # collect_debiased_data(OfficeHomeTestBed, "OfficeHome", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
            # collect_debiased_data(Office31TestBed, "Office31", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
            # collect_debiased_data(NicoTestBed, "NICO", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
            # collect_rabanser_data(CCTTestBed, "CCT", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
            # # collect_rabanser_data(OfficeHomeTestBed, "OfficeHome", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
            # collect_rabanser_data(Office31TestBed, "Office31", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
            # collect_rabanser_data(NicoTestBed, "NICO", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
            collect_rabanser_data(PolypTestBed, "Polyp", mode="normal", k=k, sampler=sampler, batch_size=batch_size)



if __name__ == '__main__':
    from features import *
    # torch.multiprocessing.set_start_method('spawn')
    collect_bias_data(5)
    collect_bias_data(0)
    collect_bias_data(-1)


    # input("next")
    # collect_data(CCTTestBed, "CCT",mode="normal")
    # collect_bias_data(5)

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
    # #
    # collect_data(Office31TestBed, "Office31", mode="normal")
    # collect_data(Office31TestBed, "Office31", mode="noise")
    # collect_data(Office31TestBed, "Office31", mode="hue")
    # collect_data(Office31TestBed, "Office31", mode="smear")
    # collect_data(Office31TestBed, "Office31", mode="saturation")
    # collect_data(Office31TestBed, "Office31", mode="brightness")
    # collect_data(Office31TestBed, "Office31", mode="contrast")
    # collect_data(Office31TestBed, "Office31", mode="multnoise")
    # collect_data(Office31TestBed, "Office31", mode="saltpepper")
    # # collect_data(Office31TestBed, "Office31", mode="fgsm")
    #
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