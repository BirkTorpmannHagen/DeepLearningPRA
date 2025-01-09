import os
from os.path import join

import numpy as np
import pandas as pd
from decimal import getcontext

from components import OODDetector, LossEstimator, Trace, SyntheticOODDetector
from rateestimators import BernoulliEstimator
from riskmodel import RiskModelWithDSD, RiskModelWithoutDSD

np.set_printoptions(precision=4, suppress=True)
getcontext().prec = 4
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option("display.max_rows", None)

# datasets
from utils import *


def load_pra_df(feature_name, batch_size=30, samples=100):
    df = pd.concat(
        [pd.read_csv(join("single_data", fname)) for fname in os.listdir("single_data") if "Polyp" in fname])
    df.drop(columns=["Unnamed: 0"], inplace=True)
    df = df[df["feature_name"] == feature_name]
    df = df[df["fold"]!="ind"]
    # df["intensity"] = df["shift"].apply(lambda x: x.split("_")[1] if "_" in x else x)
    df = df[~df["fold"].isin(["smear", "saturation", "brightness"])]
    if batch_size!=1:
        def sample_loss_feature(group, n_samples, n_size):
            samples = []
            for i in range(n_samples):
                sample = group.sample(n=n_size, replace=True)  # Sampling with replacement
                # input()
                mean_loss = sample['loss'].mean()
                mean_feature = sample['feature'].mean()
                samples.append({'loss': mean_loss, 'feature': mean_feature})
                # samples.append({'loss': mean_loss, 'feature': brunnermunzel(df[df["fold"]=="train"]["feature"], sample['feature'])[0], "reduction":"bm"})
            return pd.DataFrame(samples)
            # Return a DataFrame of means with the original group keys

        cols = list(df.columns)
        cols.remove("loss")
        cols.remove("feature")

        df = df.groupby(cols).apply(sample_loss_feature, samples, batch_size).reset_index()
        df.drop(columns=["level_2"], inplace=True)
    df["correct_prediction"] = df["loss"] < 0.5  # arbitrary threshold
    df["shift"] = df["fold"].apply(lambda x: x.split("_")[0] if "_0." in x else x)            #what kind of shift has occured?
    df["shift_intensity"] = df["fold"].apply(lambda x: x.split("_")[1] if "_" in x else x)  #what intensity?
    df["ood"] = ~df["fold"].isin(["train", "ind_val", "ind_test"])&~df["correct_prediction"] #is the prediction correct?
    return df


class SystemSimulator:
    """
    permits conditional data collection simulating model + ood detector
    """
    def __init__(self, df, maximum_loss=0.5, batch_size=30, trace_length=100, estimator=BernoulliEstimator, ood_test_shift="EndoCV2020", **kwargs):
        self.df = df
        self.maximum_loss = maximum_loss

        self.ood_test_shift = ood_test_shift
        # self.ood_detector = OODDetector(df)
        self.ood_detector = SyntheticOODDetector(kwargs["dsd_tpr"], kwargs["dsd_tnr"])
        # self.loss_estimator = LossEstimator(df)
        self.batch_size = batch_size
        dsd_tnr, dsd_tpr = self.ood_detector.get_likelihood()
        ind_ndsd_acc = self.get_conditional_prediction_likelihood_estimates("ind_val", False)
        self.risk_model = RiskModelWithDSD(dsd_tpr, dsd_tnr, ind_ndsd_acc,estimator=estimator, maximum_loss=maximum_loss)
        self.dsd_trace = Trace(trace_length)
        self.loss_trace_for_eval = Trace(trace_length)


    def sample_a_uniform_batch(self, shift="random"):
        if shift in ["ind_val", "ind_test", "ind_train"]:
            return self.df[(self.df["shift"] == shift)].sample(self.batch_size)
        else:
            return self.df[((self.df["shift"] == shift) & (self.df["ood"]))].sample(self.batch_size)


    def get_conditional_prediction_likelihood_estimates(self, shift, monitor_verdict, num_samples=1000, simulate_deployment = True):
        frame_copy = self.df[(self.df["shift"] == shift)]
        samples = []
        for i in range(num_samples):
            sample = frame_copy.sample().copy()
            if simulate_deployment:
                sample["ood"] = self.ood_detector.predict(sample)
            sample["correct"] = sample["loss"] < self.maximum_loss
            samples.append(sample)
        samples_df = pd.concat(samples)
        likelihood =  samples_df[samples_df["ood"] == monitor_verdict]["correct"].mean()
        if np.isnan(likelihood):
            return 0
        return likelihood

    def get_predictor_accuracy(self, fold, num_samples=1000):

        samples = []
        for i in range(num_samples):
            sample = self.sample_a_uniform_batch(fold)
            sample["correct"] = sample["loss"] < self.maximum_loss
            samples.append(sample)
        samples_df = pd.concat(samples)
        likelihood =  samples_df["correct"].mean()
        if np.isnan(likelihood):
            return 0
        return likelihood

    def process(self, has_shifted, index):
        shifted = has_shifted[index]
        if shifted:
            batch = self.sample_a_uniform_batch(self.ood_test_shift)
        else:
            batch = self.sample_a_uniform_batch("ind_test")

        # loss_estimate = self.loss_estimator.predict(batch)
        ood_pred = self.ood_detector.predict(batch)

        # if shifted:
        #     print(loss_estimate.mean(), batch["loss"].mean())
        # ood_pred = loss_estimate.mean() > self.maximum_loss
        # print(loss_estimate)
        # loss_estimate = 0.5
        self.loss_trace_for_eval.update(batch["loss"])
        self.dsd_trace.update(int(ood_pred))

        if index>self.dsd_trace.trace_length: #update lambda after trace length
            self.risk_model.update_rate(self.dsd_trace.trace)

        current_risk = self.risk_model.calculate_risk(self.risk_model.root)
        true_dsd_risk = self.risk_model.get_true_risk_for_sample(shifted, ood_pred, batch["loss"].mean())
        return { "Risk Estimate": current_risk, "True Risk": true_dsd_risk}

    def uniform_rate_sim(self, rate_groundtruth, num_batch_iters):
        self.risk_model.rate = rate_groundtruth
        self.risk_model.update_tree()
        has_shifted = self.risk_model.rate_estimator.sample(num_batch_iters, rate_groundtruth)
        results = []
        for i in range(num_batch_iters):
            results.append(self.process(has_shifted, index=i))
        results_df = pd.DataFrame(results)
        return results_df


if __name__ == '__main__':
    pass