# from yellowbrick.features import PCA

from testbeds import *
from utils import load_all, BATCH_SIZES, DATASETS, SYNTHETIC_SHIFTS


def compute_stats(train_features, train_losses, ind_val_features, ind_val_losses, ind_test_features, ind_test_losses, ood_features, ood_losses, fname, feature_names):
    dfs = convert_to_pandas_df(train_features, train_losses, ind_val_features, ind_val_losses, ind_test_features, ind_test_losses, ood_features, ood_losses, feature_names)
    for df, feature_name in zip(dfs, feature_names):
        df.to_csv(f"{fname}_{feature_name}.csv")

def compute_stats_no_ind(ood_features, ood_losses, fname, feature_names):
    dfs = convert_to_pandas_df_no_ind(ood_features, ood_losses, feature_names)
    for df, feature_name in zip(dfs, feature_names):
        df.to_csv(f"{fname}_{feature_name}.csv")

def collect_data(testbed_constructor, dataset_name, prefix="final_data", mode="noise"):
    print("Collecting data for", dataset_name, "in", mode, "mode")
    bench = testbed_constructor("classifier", mode=mode, batch_size=8)
    # features = [mahalanobis]
    # features = [cross_entropy,energy,knn, typicality, softmax,  grad_magnitude]
    features = [cross_entropy,energy,knn, typicality, softmax]
    # features = [knn]
    tsd = FeatureSD(bench.classifier,features)
    tsd.register_testbed(bench)
    if mode=="normal": #just compute ind and organic oods for normal mode; saves on computation time
        compute_stats(*tsd.compute_pvals_and_loss(),
                      fname=f"{prefix}/{dataset_name}_{mode}", feature_names=[f.__name__ for f in features])
    else:
        compute_stats_no_ind(*tsd.compute_pvals_and_loss(noind=True),fname=f"{prefix}/{dataset_name}_{mode}", feature_names=[f.__name__ for f in features])
    # compute_stats(*tsd.compute_pvals_and_loss(),
    #               fname=f"final_data/{dataset_name}_{mode}", feature_names=[f.__name__ for f in features])


def collect_debiased_data(testbed_constructor, dataset_name, mode="noise", sampler="RandomSampler", k=5, batch_size=8):
    # features=[cross_entropy, energy, softmax]
    features = [cross_entropy, energy, softmax, typicality, knn, grad_magnitude]
    if k!=-1:
        features.remove(knn)
    uncollected_features = features.copy()

    for feature in features:
        print(feature)
        fname = f"{dataset_name}_{mode}_{sampler}_{batch_size}_k={k}_{feature.__name__}.csv"
        if fname in os.listdir("debiased_data"):
            uncollected_features.remove(feature)
            print(f"{fname} already exists, skipping...")
    if (uncollected_features== []):
        print(f"No features left to compute for {dataset_name} in {mode} mode with {sampler} sampler and batch size {batch_size} and k={k}")
        return
    features = uncollected_features
    if k!=-1 and knn in features:
        features.remove(knn)
    print(f"Collecting data for {dataset_name} in {mode} mode with {sampler} sampler and batch size {batch_size} and k={k}")
    bench = testbed_constructor("classifier", mode=mode, sampler=sampler, batch_size=batch_size)

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
    print(f"Collecting Rabanser data for {dataset_name} in {mode} mode with {sampler} sampler and batch size {batch_size} and k={k}")
    bench = testbed_constructor("classifier", mode=mode, sampler=sampler, batch_size=batch_size)
    tsd = RabanserSD(bench.classifier,k=k)
    tsd.register_testbed(bench)
    compute_stats(*tsd.compute_pvals_and_loss(),
                  fname=f"debiased_data/{dataset_name}_{mode}_{sampler}_{batch_size}_k={k}", feature_names=["rabanser"])

def collect_knn_featurewise_data(testbed_constructor, dataset_name, mode="noise", sampler="RandomSampler", k=5, batch_size=8):
    bench = testbed_constructor("classifier", mode=mode, sampler=sampler, batch_size=batch_size)
    features = [cross_entropy, energy, softmax, typicality]
    uncollected_features = features.copy()
    for feature in features:
        fname = f"{dataset_name}_{mode}_{sampler}_{batch_size}_k={k}_{feature.__name__}.csv"
        if fname in os.listdir("debiased_data"):
            uncollected_features.remove(feature)
            print(f"{fname} already exists, skipping...")
    if (uncollected_features == []):
        print(
            f"No features left to compute for {dataset_name} in {mode} mode with {sampler} sampler and batch size {batch_size} and k={k}")
        return
    else:
        features = uncollected_features
        print(
            f"Collecting {features} data for {dataset_name} in {mode} mode with {sampler} sampler and batch size {batch_size} and k={k}")
        # features = [typicality]
        # features = [rabanser_ks]
        tsd = KNNFeaturewiseSD(bench.classifier, features, k=k)
        tsd.register_testbed(bench)
        compute_stats(*tsd.compute_pvals_and_loss(),
                      fname=f"debiased_data/{dataset_name}_{mode}_{sampler}_{batch_size}_k=featurewise_{k}",
                      feature_names=[f.__name__ for f in features])


def collect_model_wise_data(testbed_constructor, dataset_name, mode="noise"):
    for model_name in ["deeplabv3plus", "unet", "segformer"]:
        bench = testbed_constructor("classifier", mode=mode, model_name=model_name)
        # features = [mahalanobis]
        features = [cross_entropy, energy, mahalanobis, softmax, knn, typicality]

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


def collect_bias_data():
    for batch_size in BATCH_SIZES[1:]:
        for k in [0,5, 1, 10]:
        # for sampler in ["RandomSampler","ClusterSampler",  "ClassOrderSampler"]:
            for sampler in [ "RandomSampler","ClusterSampler", "SequentialSampler", "ClassOrderSampler"]:
                if sampler!="ClassOrderSampler":
                    collect_debiased_data(PolypTestBed, "Polyp", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
                    # collect_rabanser_data(PolypTestBed, "Polyp", mode="normal", k=k, sampler=sampler, batch_size=batch_size)

                collect_debiased_data(CCTTestBed, "CCT", mode="normal",k=k, sampler=sampler, batch_size=batch_size)
                collect_debiased_data(OfficeHomeTestBed, "OfficeHome", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
                collect_debiased_data(Office31TestBed, "Office31", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
                collect_debiased_data(NICOTestBed, "NICO", mode="normal", k=k, sampler=sampler, batch_size=batch_size)


def collect_single_data(testbed):
    dataset_name = testbed.__name__.split("TestBed")[0]
    if not os.path.exists(f"fine_data/{dataset_name}_normal_knn.csv"):
        print("Skipping Normal")
        collect_data(testbed, dataset_name, mode="normal", prefix="fine_data")
    for mode in ["fgsm"]:
        if os.path.exists(f"fine_data/{dataset_name}_{mode}_knn.csv"):
            continue
        print(mode)
        collect_data(testbed, dataset_name, mode=mode, prefix="fine_data")



if __name__ == '__main__':
    from features import *
    torch.multiprocessing.set_start_method('spawn')
    # collect_bias_data(-1)
    # collect_bias_data()

    # collect_data(CCTTestBed, "CCT",mode="normal")
    # collect_bias_data(5)
    collect_single_data(PolypTestBed)
    collect_single_data(OfficeHomeTestBed)
    collect_single_data(Office31TestBed)
    collect_single_data(NICOTestBed)
    collect_single_data(CCTTestBed)

    # from experiments.runtime_classification import ood_detector_correctness_prediction_accuracy
    # for batch_size in BATCH_SIZES:
    #     print(f"Running batch size {batch_size}")
    #     ood_detector_correctness_prediction_accuracy(batch_size, shift="")
    # bench = NjordTestBed(10)
    # collect_bias_data(5)
    # collect_bias_data(-1)
    # # collect_bias_data(0)
    # collect_bias_data(1)
    # collect_bias_data(10)

    # bench.split_datasets()