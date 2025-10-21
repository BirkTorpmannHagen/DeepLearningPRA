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

def collect_data(testbed_constructor, dataset_name, prefix="coarse_data", mode="noise"):
    print("Collecting data for", dataset_name, "in", mode, "mode")
    bench = testbed_constructor("classifier", mode=mode, batch_size=8)
    features = [cross_entropy,energy,knn, typicality, softmax,  grad_magnitude]
    tsd = FeatureSD(bench.classifier,features)
    tsd.register_testbed(bench)
    if mode=="normal": #just compute ind and organic oods for normal mode; saves on computation time
        compute_stats(*tsd.compute_pvals_and_loss(),
                      fname=f"{prefix}/{dataset_name}_{mode}", feature_names=[f.__name__ for f in features])
    else:
        compute_stats_no_ind(*tsd.compute_pvals_and_loss(noind=True),fname=f"{prefix}/{dataset_name}_{mode}", feature_names=[f.__name__ for f in features])
    # compute_stats(*tsd.compute_pvals_and_loss(),
    #               fname=f"coarse_data/{dataset_name}_{mode}", feature_names=[f.__name__ for f in features])


def collect_debiased_data(testbed_constructor, dataset_name, mode="noise", sampler="RandomSampler", k=5, batch_size=8):
    features = [cross_entropy, energy, softmax, typicality, knn]
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


def collect_bias_data():
    for k in [0, 5, 1, 10]:
        for batch_size in BATCH_SIZES[1:-1]:
        # for sampler in ["RandomSampler","ClusterSampler",  "ClassOrderSampler"]:
            for sampler in [ "RandomSampler","ClusterSampler", "SequentialSampler", "ClassOrderSampler"]:
                if sampler!="ClassOrderSampler":
                    collect_debiased_data(PolypTestBed, "Polyp", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
                    collect_rabanser_data(PolypTestBed, "Polyp", mode="normal", k=k, sampler=sampler, batch_size=batch_size)

                collect_debiased_data(CCTTestBed, "CCT", mode="normal",k=k, sampler=sampler, batch_size=batch_size)
                collect_rabanser_data(CCTTestBed, "CCT", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
                collect_debiased_data(OfficeHomeTestBed, "OfficeHome", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
                collect_rabanser_data(OfficeHomeTestBed, "OfficeHome", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
                collect_debiased_data(Office31TestBed, "Office31", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
                collect_rabanser_data(Office31TestBed, "Office31", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
                collect_debiased_data(NICOTestBed, "NICO", mode="normal", k=k, sampler=sampler, batch_size=batch_size)
                collect_rabanser_data(NICOTestBed, "NICO", mode="normal", k=k, sampler=sampler, batch_size=batch_size)




if __name__ == '__main__':
    from features import *
    torch.multiprocessing.set_start_method('spawn')
    # collect_bias_data(-1)
    collect_bias_data()
    for mode in ["normal"]+SYNTHETIC_SHIFTS:
        collect_data(CCTTestBed, "CCT",mode=mode)
        collect_data(OfficeHomeTestBed, "OfficeHome",mode=mode)
        collect_data(Office31TestBed, "Office31",mode=mode)
        collect_data(NICOTestBed, "NICO",mode=mode)
        collect_data(PolypTestBed, "Polyp",mode=mode)
