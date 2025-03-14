import numpy as np
import pandas as pd
from decimal import getcontext

from components import LossEstimator, Trace, SyntheticOODDetector, SplitLossEstimator, OODDetector
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


class Simulator:
    """
    Abstract simulator class
"""

    def __init__(self, df, ood_test_shift, ood_val_shift, estimator=BernoulliEstimator, trace_length=100,
                 use_synth=True, **kwargs):
        self.df = df

        self.ood_test_shift = ood_test_shift
        self.ood_val_shift = ood_val_shift
        # self.ood_detector = OODDetector(df)
        if use_synth:
            self.ood_detector = SyntheticOODDetector(kwargs["dsd_tpr"], kwargs["dsd_tnr"])
        else:
            self.ood_detector = OODDetector(df, ood_val_shift)
        self.ood_val_acc = self.get_predictor_accuracy(self.ood_val_shift)
        self.ood_test_acc = self.get_predictor_accuracy(self.ood_test_shift)
        self.ind_val_acc = self.get_predictor_accuracy("ind_val")
        self.ind_test_acc = self.get_predictor_accuracy("ind_test")
        dsd_tnr, dsd_tpr = self.ood_detector.get_likelihood()
        ind_ndsd_acc = self.get_conditional_prediction_likelihood_estimates("ind_val", False)
        ind_dsd_acc = self.get_conditional_prediction_likelihood_estimates("ind_val", True)
        ood_ndsd_acc = self.get_conditional_prediction_likelihood_estimates(ood_val_shift, False)
        ood_dsd_acc = self.get_conditional_prediction_likelihood_estimates(ood_val_shift, True)
        self.detector_tree = DetectorEventTree(dsd_tpr, dsd_tnr, ind_ndsd_acc, ind_dsd_acc, ood_ndsd_acc, ood_dsd_acc,
                                               estimator=estimator)
        self.base_tree = BaseEventTree(dsd_tpr=dsd_tpr, dsd_tnr=dsd_tnr, ood_acc=self.ood_val_acc,
                                       ind_acc=self.ind_val_acc, estimator=estimator)
        self.dsd_trace = Trace(trace_length)
        self.loss_trace_for_eval = Trace(trace_length)
        self.shifts = self.df[self.df["ood"]]["shift"].unique()

    def get_conditional_prediction_likelihood_estimates(self, shift, monitor_verdict, num_samples=1000):
        frame_copy = self.df[(self.df["shift"] == shift)]
        samples = []
        for i in range(num_samples):
            sample = frame_copy.sample(replace=True).copy()
            sample["ood_pred"] = self.ood_detector.predict(sample)
            samples.append(sample)
        samples_df = pd.concat(samples)
        # get the likelihood of correct prediction given that the detectorr predicts monitor_verdict
        likelihood = samples_df[samples_df["ood_pred"] == monitor_verdict]["correct_prediction"].mean()

        if np.isnan(likelihood):
            return 0
        return likelihood

    def get_predictor_accuracy(self, fold):
        return self.df[self.df["shift"] == fold]["correct_prediction"].mean()



    def sim(self, rate_groundtruth, num_batch_iters):
        self.detector_tree.rate = rate_groundtruth
        self.detector_tree.update_tree()
        has_shifted = self.detector_tree.rate_estimator.sample(num_batch_iters, rate_groundtruth)
        results = []
        results_trace = []
        for i in range(num_batch_iters):
            current_horizon_results = self.process(has_shifted, index=i)
            if current_horizon_results is not None:
                results.append(current_horizon_results)
                if len(results) > self.dsd_trace.trace_length:
                    df = pd.concat(
                        results[-self.dsd_trace.trace_length:])  # get the data corresponding to the last trace length
                    results_trace.append(df.groupby(["Tree"]).mean().reset_index())
        results_df = pd.concat(results_trace)
        results_df["Rate Error"] = np.abs(results_df["Estimated Rate"] - rate_groundtruth)
        results_df["Accuracy Error"] = np.abs(results_df["E[f(x)=y]"] - results_df["Accuracy"])
        return results_df


class UniformBatchSimulator(Simulator):
    """
    permits conditional data collection simulating model + ood detector
    """
    def __init__(self, df, ood_test_shift, ood_val_shift,  estimator=BernoulliEstimator, trace_length=100, use_synth=True, **kwargs):
        super().__init__(df, ood_test_shift, ood_val_shift, estimator, trace_length, use_synth, **kwargs)


    def sample_a_uniform_batch(self, shift):
        return self.df[self.df["shift"] == shift].sample()

    def process(self, has_shifted, index):
        shifted = has_shifted[index]
        if shifted:
            batch = self.sample_a_uniform_batch(self.ood_test_shift)
        else:
            batch = self.sample_a_uniform_batch("ind_test")
        ood_pred = self.ood_detector.predict(batch)
        batch["ood_pred"] = ood_pred #todo, this is a hack
        # self.loss_trace_for_eval.update(batch["loss"])
        self.dsd_trace.update(int(ood_pred))

        if index>self.dsd_trace.trace_length: #update lambda after trace length
            self.detector_tree.update_rate(self.dsd_trace.trace)
            self.base_tree.update_rate(self.dsd_trace.trace)

            current_risk = self.detector_tree.calculate_risk(self.detector_tree.root)
            current_expected_accuracy = self.detector_tree.calculate_expected_accuracy(self.detector_tree.root)
            true_dsd_risk = self.detector_tree.get_true_risk_for_sample(batch)

            current_base_risk = self.base_tree.calculate_risk(self.base_tree.root)
            current_base_expected_accuracy = self.base_tree.calculate_expected_accuracy(self.base_tree.root)
            true_base_risk = self.base_tree.get_true_risk_for_sample(batch)
            accuracy = batch["correct_prediction"].mean()
            # accuracy = self.ind_test_acc
            data = pd.DataFrame( {"t":[index, index], "Tree": ["Detector Tree", "Base Tree"], "Risk Estimate": [current_risk, current_base_risk],
                                           "True Risk": [true_dsd_risk, true_base_risk], "E[f(x)=y]":[current_expected_accuracy, current_base_expected_accuracy],
                                           "Accuracy": [accuracy, accuracy], "ood_pred": [ood_pred, ood_pred], "is_ood": [shifted, shifted],
                                           "Estimated Rate":[self.detector_tree.rate, self.base_tree.rate],
                 "ind_acc": [self.ind_val_acc, self.ind_val_acc], "ood_val_acc": [self.ood_val_acc, self.ood_val_acc], "ood_test_acc": [self.ood_test_acc, self.ood_test_acc]})

            return data


    def sim(self, rate_groundtruth, num_batch_iters):
        self.detector_tree.rate = rate_groundtruth
        self.detector_tree.update_tree()
        has_shifted = self.detector_tree.rate_estimator.sample(num_batch_iters, rate_groundtruth)
        results = []
        results_trace = []
        for i in range(num_batch_iters):
            current_horizon_results = self.process(has_shifted, index=i)
            if current_horizon_results is not None:
                results.append(current_horizon_results)
                if len(results)>self.dsd_trace.trace_length:

                    df = pd.concat(results[-self.dsd_trace.trace_length:]) #get the data corresponding to the last trace length
                    results_trace.append(df.groupby(["Tree"]).mean().reset_index())
        results_df = pd.concat(results_trace)
        results_df["Rate Error"] = np.abs(results_df["Estimated Rate"] - rate_groundtruth)
        results_df["Accuracy Error"] = np.abs(results_df["E[f(x)=y]"] - results_df["Accuracy"])
        return results_df





if __name__ == '__main__':
    pass