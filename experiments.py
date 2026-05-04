import warnings

from experiments.pra import collect_re_accuracy_estimation_data

warnings.filterwarnings("ignore")
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
    error_heatmap()                                        # Figure 2
    error_per_accuracy()                                   # Figure 3
    dr_gap_correlation_distribution(1, pretrain=True)      # NEW: addresses cherry-picking
    threshold_method_comparison(1, pretrain=True)          # NEW: addresses ID-only thresholding
    ood_detector_ba()


if __name__ == '__main__':
    run_acc_prediction_experiments()
