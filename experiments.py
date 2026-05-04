import warnings

from experiments.pra import collect_re_accuracy_estimation_data

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")  # non-interactive: write PDFs, don't open windows
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False  # local TeX install lacks cmss10.tfm; mathtext is fine for figure prep
from experiments.accuracy_prediction import *

np.set_printoptions(precision=3, suppress=True)


def ood_detector_ba():
    df = get_all_ood_detector_data(1, filter_organic=True, filter_best=True, pretrain=True)
    print(df.columns)
    cols = [c for c in ["DR", "BA", "Accuracy"] if c in df.columns]
    print(df.groupby(["Dataset", "Fold"])[cols].mean())


def run_acc_prediction_experiments():
    test_generalization_gap_estimation(1, pretrain=True)   # Figure 1
    error_per_accuracy()                                   # Figure 3
    dr_gap_correlation_distribution(1, pretrain=True)      # robustness of DR-gap relationship
    threshold_method_comparison(1, pretrain=True)          # ID-only thresholding viability
    atc_comparison(1, pretrain=True)                       # ATC head-to-head, per-fold

    import pandas as pd
    p1 = loo_fold_comparison(1, pretrain=True, anchor=False)
    p2 = loo_fold_comparison(1, pretrain=True, anchor=True)
    if not p1.empty and not p2.empty:
        combined = pd.concat([p1, p2[["Ours-anchored"]]], axis=1)
        combined = combined.reindex(columns=["Ours", "Ours-anchored", "ATC-MC", "ATC-NE"])
        combined.to_csv("figures/loo_fold_comparison.csv")
        print("\n=== LOO per-fold MAE (saved to figures/loo_fold_comparison.csv) ===")
        print(combined.round(4).to_string())

    intensity_breakdown_plot(1, pretrain=True)             # MAE vs shift intensity, per shift type

    ood_detector_ba()


if __name__ == '__main__':
    run_acc_prediction_experiments()
