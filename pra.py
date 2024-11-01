import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LinearGAM
from decimal import Decimal, getcontext
np.set_printoptions(precision=4, suppress=True)
getcontext().prec = 4
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from scipy.stats import beta

MISDIAGNOSIS = 6100
UNNECESSARY_INTERVENTION = 635+100+0.2*MISDIAGNOSIS
NECESSARY_INTERVENTION = 635+100+0.2*MISDIAGNOSIS #arbitrary, but lower than unnecessary intervention
CORRECT_DIAGNOSIS = 635 # cost of correct diagnosis during AI screening



def load_pra_df(feature_name, batch_size=30, samples=100):
    df = pd.concat(
        [pd.read_csv(join("single_data", fname)) for fname in os.listdir("single_data") if "Polyp" in fname])
    df.drop(columns=["Unnamed: 0"], inplace=True)
    df = df[df["feature_name"] == feature_name]
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
    df["shift"] = df["fold"].apply(lambda x: x.split("_")[0] if "_" in x else x)
    df["shift_intensity"] = df["fold"].apply(lambda x: x.split("_")[1] if "_" in x else x)
    # print(df.groupby("fold").head(10))
    # input()
    return df


def get_beta_binomial(rate, num_iters):
    """ returns a list of beta_binomial-distributed events"""
    return np.random.binomial(1, rate, num_iters)

# class ShiftLikelihoodEstimator:
#     def __init__(self):
#


class OODDetector:
    def __init__(self, df):
        self.df = df
        threshold_df = df[(df["fold"] == "ind")]

        self.higher_is_ood = df[df["fold"]=="ood"]["feature"].mean() > df[df["fold"]=="ind"]["feature"].mean()
        if self.higher_is_ood:
            self.threshold = threshold_df["feature"].sample(len(threshold_df)//2).max() # sample half of the data to serve as thresholding
        else:
            self.threshold = threshold_df["feature"].sample(len(threshold_df)//2).min()

    def predict(self, batch):
        if self.higher_is_ood:
            return batch["feature"].mean() > self.threshold
        else:
            return batch["feature"].mean() < self.threshold

    def get_likelihood(self):
        self.ind = self.df[self.df["fold"] == "ind"]
        self.ood = self.df[~self.df["fold"].isin(["ind", "train"])]
        tpr = (self.ood["feature"] > self.threshold).mean()
        tnr = (self.ind["feature"] < self.threshold).mean()
        return tpr, tnr


class LossEstimator:
    def __init__(self, df):
        self.df = df
        self.gam = LinearGAM(constraints="monotonic_inc")
        self.train_set = df[(df["fold"]=="ind")|df["fold"].str.contains("noise_")]
        self.train(df["feature"], df["loss"])
        self.plot_gam()

    def plot_gam(self):
        plt.figure()
        plt.scatter(self.train_set["feature"], self.train_set["loss"], label="train")
        x = np.linspace(self.train_set["feature"].min(), self.train_set["feature"].max(), 100)
        plt.plot(x, self.gam.predict(x), label="gam")
        plt.legend()
        plt.show()
    def train(self, X, Y):
        self.gam.fit(X, Y)
        return np.sqrt(((self.gam.predict(X)-Y)**2).mean()) #MSE

    def evaluate(self, batch):
        return np.sqrt(((self.gam.predict(batch["feature"])-batch["loss"])**2).mean())

    def predict(self, batch):
        return self.gam.predict(batch["feature"])



class Trace:
    """
    Container for dsd verdict traces; used to estimate lambda
    """
    def __init__(self, trace_length=100):
        self.trace_length = trace_length
        self.trace = []

    def update(self, item):
        if len(self.trace) == self.trace_length:
            self.trace.pop(0)
        self.trace.append(item)
        return self.trace


class SystemSimulator:
    """
    permits conditional data collection simulating model + ood detector
    """
    def __init__(self, df, minimum_iou=0.5, batch_size=30, trace_length=100, prior_alpha=1, prior_beta=9, use_estimators=False):
        self.df = df
        self.ood_detector = OODDetector(df)
        self.loss_estimator = LossEstimator(df)
        dsd_tnr, dsd_tpr = self.ood_detector.get_likelihood()
        ind_ndsd_acc = self.get_conditional_prediction_likelihood_estimates("ind", False)
        self.risk_model = RiskModelWithDSD(dsd_tpr, dsd_tnr, ind_ndsd_acc, prior_alpha=prior_alpha, prior_beta=prior_beta, use_estimators=use_estimators)
        ind_acc = self.get_predictor_accuracy("ind")
        ood_acc = self.get_predictor_accuracy("ood")

        self.risk_model_without_dsd = RiskModelWithoutDSD(1, 0.9, ood_acc, ind_acc, use_estimators)
        self.minimum_iou = minimum_iou
        self.dsd_trace = Trace(trace_length)
        self.loss_trace_for_eval = Trace(trace_length)
        self.use_estimators = use_estimators


    def sample_a_uniform_batch(self, shift):
        if shift=="ind":
            return self.df[(self.df["fold"] == "ind")].sample()
        return self.df[((self.df["shift"] == shift) & (self.df["loss"]>0.5))].sample()


    def get_conditional_prediction_likelihood_estimates(self, fold, monitor_verdict, num_samples=1000, simulate_deployment = True):
        frame_copy = self.df[(self.df["fold"] == fold)]
        samples = []
        for i in range(num_samples):
            sample = frame_copy.sample()
            if simulate_deployment:
                sample["ood"] = self.ood_detector.predict(sample)
            else:
                sample["ood"] = frame_copy["fold"] != "ind"
            sample["correct"] = sample["loss"] < 0.5
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
            sample["correct"] = sample["loss"] < 0.5
            samples.append(sample)
        samples_df = pd.concat(samples)
        likelihood =  samples_df["correct"].mean()
        if np.isnan(likelihood):
            return 0
        return likelihood


    def get_acc_given_dsd_metrics(self):
        pass

    def estimate_consequence(self, batch):
        return self.loss_estimator.predict(batch)

    def process(self, has_shifted, index):
        shifted = has_shifted[index]
        if shifted:
            shifts = self.df["shift"].unique()
            shift = np.random.choice(shifts[(shifts != "ind")&(shifts != "train")])
            batch = self.sample_a_uniform_batch(shift)
            # batch = self.sample_a_uniform_batch("ood")
        else:
            batch = self.sample_a_uniform_batch("ind")

        ood_pred = self.ood_detector.predict(batch)
        loss_estimate = self.loss_estimator.evaluate(batch)
        # loss_estimate = 0.5
        self.loss_trace_for_eval.update(batch["loss"])
        self.dsd_trace.update(int(ood_pred))


        if index>self.dsd_trace.trace_length: #update lambda after trace length
            self.risk_model.update_rate(self.dsd_trace.trace)
            self.risk_model_without_dsd.update_rate(self.dsd_trace.trace)

        correct = ood_pred  == shifted
        current_risk = self.risk_model.calculate_risk(self.risk_model.root)
        true_dsd_risk = self.risk_model.get_true_risk_for_sample(shifted, ood_pred, batch["loss"].mean())

        # no_dsd_risk = 0
        no_dsd_risk = self.risk_model_without_dsd.calculate_risk(self.risk_model_without_dsd.root)
        true_nodsd_risk = self.risk_model_without_dsd.get_true_risk_for_sample(shifted, batch["loss"].mean())
        true_loss = np.mean(self.loss_trace_for_eval.trace)
        return { "Risk Estimate": current_risk, "No DSD Risk Estimate": no_dsd_risk, "True DSD Risk": true_dsd_risk, "True No DSD risk": true_nodsd_risk}

    def uniform_rate_sim(self, rate_groundtruth, num_batch_iters):
        if not self.use_estimators:
            self.risk_model.rate = rate_groundtruth
            self.risk_model.update_tree()
            self.risk_model_without_dsd.rate = rate_groundtruth
            self.risk_model_without_dsd.update_tree()
        has_shifted = get_beta_binomial(rate_groundtruth, num_batch_iters)
        results = []
        for i in range(num_batch_iters):
            results.append(self.process(has_shifted, index=i))
        results_df = pd.DataFrame(results)
        return results_df




class RiskNode:
    def __init__(self, probability, consequence=0, left=None, right=None):
        self.left = None
        self.right = None
        self.probability = probability #probability of right child
        self.consequence = 0 #if non leaf node

    def is_leaf(self):
        return self.left is None and self.right is None

class RiskModel:
    def __init__(self, prior_alpha=1, prior_beta=0.9, use_estimators=True):
        """
                Binary Tree defined by the following structure:
                ood+/- -> dsd+/- -> prediction+/- -> consequence
                """
        self.alpha = prior_alpha
        self.beta = prior_beta
        prior_dist = beta(self.alpha, self.beta)
        self.rate = prior_dist.mean()
        self.use_estimators = use_estimators
        self.root = RiskNode(1)  # root
        # self.update_tree()

    def update_tree(self):
        raise NotImplementedError

    def update_rate(self, trace):
        if self.use_estimators:
            shift_counts = np.sum(trace)
            non_shift_counts = (len(trace) - shift_counts)
            self.alpha = self.alpha + shift_counts
            self.beta = self.beta + non_shift_counts
            dist = beta(self.alpha, self.beta)
            posterior_mean = dist.mean()
            self.rate = posterior_mean
        self.update_tree()

    def calculate_risk(self, node, accumulated_prob=1.0):
        if node is None:
            return 0
        # Multiply the probability of the current node with the accumulated probability so far
        current_prob = accumulated_prob * node.probability

        # If it's a leaf node, calculate risk as probability * consequence
        if node.is_leaf():
            return current_prob * node.consequence

        # Recursively calculate risk for left and right children
        left_risk = self.calculate_risk(node.left, current_prob)
        right_risk = self.calculate_risk(node.right, current_prob)

        # Total risk is the sum of risks from both branches
        return left_risk + right_risk


class RiskModelWithDSD(RiskModel):
    def __init__(self, dsd_tpr, dsd_tnr, ind_ndsd_acc, prior_alpha=1, prior_beta=0.9, use_estimators=False):
        """
                Binary Tree defined by the following structure:
                ood+/- -> dsd+/- -> prediction+/- -> consequence
                """
        super().__init__(prior_alpha, prior_beta, use_estimators)
        self.dsd_tpr, self.dsd_tnr = dsd_tpr, dsd_tnr
        self.ind_ndsd_acc  = ind_ndsd_acc
        print(f"Initializing Risk Model with {dsd_tpr, dsd_tnr, ind_ndsd_acc}")
        self.root = RiskNode(1) #root
        self.update_tree()
        # self.print_tree()

    def get_true_risk_for_sample(self, is_ood, detected_as_ood, loss):
        if is_ood:
            if detected_as_ood:
                return NECESSARY_INTERVENTION
            else:
                return MISDIAGNOSIS
        else:
            if detected_as_ood:
                return UNNECESSARY_INTERVENTION
            else:
                return CORRECT_DIAGNOSIS if loss < 0.5 else MISDIAGNOSIS

    def update_tree(self):
        self.root.left = RiskNode(1-self.rate) #data is ind
        self.root.right = RiskNode(self.rate) #data is ood

        self.root.left.left = RiskNode(self.dsd_tnr) #data is ind, dsd predicts ind
        self.root.left.right = RiskNode(1-self.dsd_tnr) #data is ind, dsd predicts ood
        self.root.right.left = RiskNode(1-self.dsd_tpr) #data is ood, dsd predicts ind
        self.root.right.right = RiskNode(self.dsd_tpr) #data is ood, dsd predicts ood

        #dsd consequences
        self.root.left.right.consquence = UNNECESSARY_INTERVENTION #data is ind, dsd predicts ood
        self.root.right.left.consequence = MISDIAGNOSIS #data is ood, dsd predicts ind
        self.root.right.right.consequence = NECESSARY_INTERVENTION #data is ood, dsd predicts ood

        #dsd predicts ind, accuracy of model
        self.root.left.left.left = RiskNode(self.ind_ndsd_acc) #data is ind, dsd predicts ind, prediction is correct
        self.root.left.left.right = RiskNode(1-self.ind_ndsd_acc) #data is ind, dsd predicts ind, prediction is incorrect

        self.root.left.left.left.consequence = CORRECT_DIAGNOSIS  # data is ind, dsd predicts ind, prediction is correct (no intervention)
        self.root.left.left.right.consequence = MISDIAGNOSIS  # data is ind, dsd predicts ind, prediction is incorrect (loss)


    def print_tree(self):
        print("\t\t\t\tRoot\t\t\t\t")
        print(f"{self.root.left.probability}*{self.root.left.consequence} \t\t\t {self.root.right.probability}*{self.root.right.consequence}")
        print(f"{self.root.left.left.probability}*{self.root.left.left.consequence} \t\t\t {self.root.left.right.probability}*{self.root.left.right.consequence}\t\t\t{self.root.right.left.probability}*{self.root.right.left.consequence}\t\t\t{self.root.right.right.probability}*{self.root.right.right.consequence}")
        print(f"{self.root.left.left.left.probability}*{self.root.left.left.left.consequence} \t\t\t {self.root.left.left.right.probability}*{self.root.left.left.right.consequence}")

class RiskModelWithoutDSD(RiskModel):
    def __init__(self, prior_alpha, prior_beta, ood_acc, ind_acc, use_estimators=False):
        super().__init__(prior_alpha, prior_beta, use_estimators)
        self.ood_acc = ood_acc
        self.ind_acc = ind_acc
        self.root = RiskNode(1)
        self.update_tree()

    def get_true_risk_for_sample(self, is_ood, loss):
        if is_ood:
            return MISDIAGNOSIS
        else:
            return CORRECT_DIAGNOSIS if loss < 0.5 else MISDIAGNOSIS
    def update_tree(self):

        self.root.left = RiskNode(1-self.rate)
        self.root.right = RiskNode(self.rate)
        self.root.left.left = RiskNode(self.ind_acc) #data is ind, prediction is correct
        self.root.left.right = RiskNode(1-self.ind_acc) #data is ind, prediction is incorrect
        self.root.right.left = RiskNode(self.ood_acc)
        self.root.right.right = RiskNode(1-self.ood_acc)
        self.root.left.left.consequence = CORRECT_DIAGNOSIS
        self.root.left.right.consequence = MISDIAGNOSIS
        self.root.right.left.consequence = CORRECT_DIAGNOSIS
        self.root.right.right.consequence = MISDIAGNOSIS


if __name__ == '__main__':
    data = load_pra_df("knn", batch_size=10, samples=1000)
    print(NECESSARY_INTERVENTION)
    sim = SystemSimulator(data, use_estimators=True)
    results = sim.uniform_rate_sim(0.5, 10000)
    print(results.mean())
    # x = np.linspace(0,1,11)
    # dsd_r = []
    # ndsd_r = []
    # for i in x:
    #     sim = SystemSimulator(data)
    #     results = sim.uniform_rate_sim(i, 10000)
    #     dsd_r.append(results[0])
    #     ndsd_r.append(results[1])
    # plt.plot(x, dsd_r, label="AI with shift detection")
    # plt.plot(x, ndsd_r, label="AI without shift detection")
    # plt.axhline(NECESSARY_INTERVENTION, color="red", label="Regular clinical screening")
    # plt.xlabel("Shift Rate")
    # plt.ylabel("Expected per-patient cost ($)")
    # plt.legend()
    # plt.title("Cost of AI Screening with and without Shift Detection")
    # print(f"Avg Risk: {np.mean(dsd_r)}\t Avg Risk (No DSD): {np.mean(ndsd_r)}, Current Cost: {NECESSARY_INTERVENTION}")
    #
    # plt.show()
