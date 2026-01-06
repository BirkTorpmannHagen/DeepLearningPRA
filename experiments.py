# from albumentations.random_utils import normal

import warnings
from experiments.pra import collect_re_accuracy_estimation_data

warnings.filterwarnings("ignore")
from experiments.accuracy_prediction import *

def xval_errors(values):
    return np.mean([np.sum(np.abs(np.subtract.outer(valwise_accuracies, valwise_accuracies))) / np.sum(
        np.ones_like(valwise_accuracies) - np.eye(valwise_accuracies.shape[0])) for valwise_accuracies in values])

# pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(precision=3, suppress=True)


def get_dsd_verdicts_given_true_trace(trace, tpr, tnr):
    def transform(v):
        if v==1:
            if np.random.rand() < tpr:
                return 1
            else:
                return 0
        else:
            if np.random.rand() < tnr:
                return 0
            else:
                return 1
    return [transform(i) for i in trace]




def ood_detector_ba():
    df = get_all_ood_detector_data(1, filter_organic=True, filter_best=True, pretrain=True)
    print(df.columns)
    print(df.groupby(["Dataset", "Fold"])[["DR", "BA", "Accuracy"]].mean())


def run_acc_prediction_experiments():
    # test_generalization_gap_estimation(1, pretrain=True)
    # for model in MODELS:
    # get_acc_prediction_results(1, pretrain=True)
    # ood_detector_correctness_prediction_accuracy(1, model="deeplabv3plus", shift="")

    # acc_prediction_table(pretrain=True)
    # collect_re_accuracy_estimation_data()
    # get_all_pre_data()
    # error_heatmap()
    # error_per_accuracy()
    ood_detector_ba()



if __name__ == '__main__':
    run_acc_prediction_experiments()



