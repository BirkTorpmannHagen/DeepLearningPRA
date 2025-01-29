import numpy as np
import pandas as pd
from decimal import getcontext

from watchdog.observers.inotify_c import inotify_init

from components import LossEstimator, Trace, SyntheticOODDetector, SplitLossEstimator
from rateestimators import BernoulliEstimator
from riskmodel import DetectorEventTree, BaseEventTree

np.set_printoptions(precision=4, suppress=True)
getcontext().prec = 4
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option("display.max_rows", None)

# datasets
from utils import *


class SystemSimulator:
    """
    permits conditional data collection simulating model + ood detector
    """
    def __init__(self, df, maximum_loss=0.5, batch_size=30, trace_length=100, estimator=BernoulliEstimator, ood_test_shift=ENDOCV, ood_val_shift=CVCCLINIC, **kwargs):
        self.df = df
        self.maximum_loss = maximum_loss

        self.ood_test_shift = ood_test_shift
        self.ood_val_shift = ood_val_shift
        # self.ood_detector = OODDetector(df)
        self.ood_detector = SyntheticOODDetector(kwargs["dsd_tpr"], kwargs["dsd_tnr"])
        self.batch_size = batch_size
        self.ood_val_acc = self.get_predictor_accuracy(self.ood_val_shift)
        self.ood_test_acc = self.get_predictor_accuracy(self.ood_test_shift)
        self.ind_val_acc = self.get_predictor_accuracy("ind_val")
        dsd_tnr, dsd_tpr = self.ood_detector.get_likelihood()
        ind_ndsd_acc = self.get_conditional_prediction_likelihood_estimates("ind_val", False)
        ind_dsd_acc = self.get_conditional_prediction_likelihood_estimates("ind_val", True)
        ood_ndsd_acc = self.get_conditional_prediction_likelihood_estimates(ood_val_shift, False)
        ood_dsd_acc = self.get_conditional_prediction_likelihood_estimates(ood_val_shift, True)
        print(ind_ndsd_acc, ind_dsd_acc, ood_ndsd_acc, ood_dsd_acc)
        self.detector_tree = DetectorEventTree(dsd_tpr, dsd_tnr, ind_ndsd_acc, ind_dsd_acc, ood_ndsd_acc, ood_dsd_acc, estimator=estimator, maximum_loss=maximum_loss)
        self.base_tree = BaseEventTree(dsd_tpr=dsd_tpr, dsd_tnr=dsd_tnr, ood_acc = self.ood_val_acc, ind_acc=self.ind_val_acc, maximum_loss=maximum_loss, estimator=estimator)
        self.dsd_trace = Trace(trace_length)
        self.loss_trace_for_eval = Trace(trace_length)
        self.shifts = self.df[self.df["ood"]]["shift"].unique()


    def sample_a_uniform_batch(self, shift="random"):
        if shift in ["ind_val", "ind_test", "ind_train"]:
            return self.df[(self.df["shift"] == shift)].sample(self.batch_size)
        else:
            if shift == "random":
                shift = np.random.choice(self.shifts)
                # print(shift)
            return self.df[self.df["shift"] == shift].sample(self.batch_size, replace=True)


    def get_conditional_prediction_likelihood_estimates(self, shift, monitor_verdict, num_samples=1000):
        frame_copy = self.df[(self.df["shift"] == shift)]
        # print(frame_copy)
        # input()
        samples = []
        for i in range(num_samples):
            sample = frame_copy.sample().copy()
            sample["ood_pred"] = self.ood_detector.predict(sample)
            sample["correct"] = sample["loss"] < self.maximum_loss
            samples.append(sample)
        samples_df = pd.concat(samples)
        #get the likelihood of correct prediction given that the detectorr predicts monitor_verdict
        likelihood =  samples_df[samples_df["ood_pred"] == monitor_verdict]["correct"].mean()

        if np.isnan(likelihood):
            return 0
        return likelihood

    def get_predictor_accuracy(self, fold, num_samples=1000):
        return self.df[self.df["fold"]==fold]["correct_prediction"].mean()


    def process(self, has_shifted, index):
        shifted = has_shifted[index]
        if shifted:
            batch = self.sample_a_uniform_batch(self.ood_test_shift)
            # batch = self.sample_a_uniform_batch("random")
        else:
            batch = self.sample_a_uniform_batch("ind_test")
        ood_pred = self.ood_detector.predict(batch)
        self.loss_trace_for_eval.update(batch["loss"])
        self.dsd_trace.update(int(ood_pred))

        if index>self.dsd_trace.trace_length: #update lambda after trace length
            self.detector_tree.update_rate(self.dsd_trace.trace)
            self.base_tree.update_rate(self.dsd_trace.trace)
        batch_loss = batch["loss"].mean()
        current_risk = self.detector_tree.calculate_risk(self.detector_tree.root)
        current_expected_accuracy = self.detector_tree.calculate_expected_accuracy(self.detector_tree.root)
        true_dsd_risk = self.detector_tree.get_true_risk_for_sample(shifted, ood_pred, batch_loss)

        current_base_risk = self.base_tree.calculate_risk(self.base_tree.root)
        current_base_expected_accuracy = self.base_tree.calculate_expected_accuracy(self.base_tree.root)
        true_base_risk = self.base_tree.get_true_risk_for_sample(batch_loss)
        accuracy = (batch["loss"] < self.maximum_loss).mean()
        return [
            {"Tree": "Detector Tree", "Risk Estimate": current_risk, "True Risk": true_dsd_risk,
              "E[f(x)=y]":current_expected_accuracy, "Accuracy": accuracy,
             "ood_pred": ood_pred, "is_ood": shifted, "Estimated Rate":self.detector_tree.rate,
             "ind_acc": self.ind_val_acc, "ood_val_acc": self.ood_val_acc, "ood_test_acc": self.ood_test_acc},
            {"Tree": "Base Tree", "Risk Estimate": current_base_risk, "True Risk": true_base_risk,
                "E[f(x)=y]":current_base_expected_accuracy, "Accuracy": accuracy,
                 "ood_pred": ood_pred, "is_ood": shifted, "Estimated Rate":self.base_tree.rate,
                 "ind_acc": self.ind_val_acc, "ood_val_acc": self.ood_val_acc, "ood_test_acc": self.ood_test_acc}
        ]


    def uniform_rate_sim(self, rate_groundtruth, num_batch_iters):
        self.detector_tree.rate = rate_groundtruth
        self.detector_tree.update_tree()
        has_shifted = self.detector_tree.rate_estimator.sample(num_batch_iters, rate_groundtruth)
        results = []
        for i in range(num_batch_iters):
            detector_tree_results, base_tree_results = self.process(has_shifted, index=i)
            results.append(detector_tree_results)
            results.append(base_tree_results)
        results_df = pd.DataFrame(results)
        return results_df


if __name__ == '__main__':
    pass