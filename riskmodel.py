import numpy as np

from components import LossEstimator
from rateestimators import BernoulliEstimator

MISDIAGNOSIS = 635+6100
UNNECESSARY_INTERVENTION = 635+0.2*MISDIAGNOSIS
NECESSARY_INTERVENTION = 635+0.2*MISDIAGNOSIS - 50#arbitrary, but lower than unnecessary intervention
CORRECT_DIAGNOSIS = 635 # cost of correct diagnosis during AI screening


class RiskNode:
    def __init__(self, probability, consequence=0, left=None, right=None):
        self.left = None
        self.right = None
        self.probability = probability #probability of right child
        self.consequence = 0 #if non leaf node
        self.corresponds_to_correct_prediction = False

    def is_leaf(self):
        return self.left is None and self.right is None


class RiskModel:
    def __init__(self, estimator = BernoulliEstimator):
        """
                Binary Tree defined by the following structure:
                ood+/- -> dsd+/- -> prediction+/- -> consequence
                """



        self.root = RiskNode(1)  # root
        self.rate_estimator = estimator()
        self.rate = self.rate_estimator.get_rate()
        # self.update_tree()

    def update_tree(self):
        raise NotImplementedError

    def update_rate(self, trace):
        self.rate = self.rate_estimator.update(trace)
        self.update_tree()

    def get_risk_estimate(self):
        return self.calculate_risk(self.root)

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

    def calculate_expected_accuracy(self, node, accumulated_prob=1.0):
        if node is None:
            return 0
        # Multiply the probability of the current node with the accumulated probability so far
        current_prob = accumulated_prob * node.probability

        if node.is_leaf():
            if node.corresponds_to_correct_prediction:
                return current_prob
            else:
                return 0

        # Recursively calculate risk for left and right children
        left_risk = self.calculate_expected_accuracy(node.left, current_prob)
        right_risk = self.calculate_expected_accuracy(node.right, current_prob)
        # Total risk is the sum of risks from both branches
        return left_risk + right_risk


class DetectorEventTree(RiskModel):
    def __init__(self, dsd_tpr, dsd_tnr, ind_ndsd_acc, ind_dsd_acc, ood_ndsd_acc, ood_dsd_acc, estimator):
        """
                Binary Tree defined by the following structure:
                ood+/- -> dsd+/- -> prediction+/- -> consequence
                """
        super().__init__(estimator)
        self.dsd_tpr, self.dsd_tnr = dsd_tpr, dsd_tnr
        self.ind_ndsd_acc  = ind_ndsd_acc
        self.ood_ndsd_acc = ood_ndsd_acc
        self.ind_dsd_acc = ind_dsd_acc
        self.ood_dsd_acc = ood_dsd_acc
        self.rate_estimator.update_tpr_tnr(dsd_tpr, dsd_tnr) #todo, dumb shortcut
        self.root = RiskNode(1) #root
        self.update_tree()
        # self.print_tree()

        # self.print_tree()

    def get_true_risk_for_sample(self, data):
        assert len(data) == 1
        is_ood = data["ood"].all()
        detected_as_ood = data["ood_pred"].all()
        correct = data["correct_prediction"].all()
        if is_ood:
            if detected_as_ood:
                if correct:
                    return UNNECESSARY_INTERVENTION
                else:
                    return NECESSARY_INTERVENTION
            elif not detected_as_ood:
                if correct:
                    return CORRECT_DIAGNOSIS
                else:
                    return MISDIAGNOSIS
        else:
            if detected_as_ood:
                if correct:
                    return UNNECESSARY_INTERVENTION
                else:
                    return NECESSARY_INTERVENTION
            elif not detected_as_ood:
                return CORRECT_DIAGNOSIS if correct else MISDIAGNOSIS

    def update_tree(self):
        self.root.left = RiskNode(1-self.rate) #data is ind
        self.root.right = RiskNode(self.rate) #data is ood

        self.root.left.left = RiskNode(self.dsd_tnr) #data is ind, dsd predicts ind
        self.root.left.right = RiskNode(1-self.dsd_tnr) #data is ind, dsd predicts ood
        self.root.right.left = RiskNode(1-self.dsd_tpr) #data is ood, dsd predicts ind
        self.root.right.right = RiskNode(self.dsd_tpr) #data is ood, dsd predicts ood

        self.root.right.left.left = RiskNode(self.ood_ndsd_acc) #data is ood, dsd predicts ind, prediciton is correct
        self.root.right.left.left.corresponds_to_correct_prediction = True
        self.root.right.left.right = RiskNode(1-self.ood_ndsd_acc) #data is ood, dsd predicts ind, predictions is incorrect
        self.root.right.right.left = RiskNode(self.ood_dsd_acc)
        self.root.right.right.left.corresponds_to_correct_prediction = True
        self.root.right.right.right = RiskNode(1-self.ood_dsd_acc)

        self.root.left.left.left = RiskNode(self.ind_ndsd_acc) #data is ind, dsd predicts ind, prediction is correct
        self.root.left.left.left.corresponds_to_correct_prediction = True
        self.root.left.left.right = RiskNode(1-self.ind_ndsd_acc) #data is ind, dsd predicts ind, prediction is incorrect
        self.root.left.right.left = RiskNode(self.ind_dsd_acc)
        self.root.left.right.left.corresponds_to_correct_prediction = True
        self.root.left.right.right = RiskNode(1-self.ind_dsd_acc)

        #note that no further branching is needed here, since the outcome is deterministic if dsd predicts ood

        self.root.left.right.left.consequence = UNNECESSARY_INTERVENTION #data is ind, dsd predicts ood, prediction is correct
        self.root.left.right.right.consequence = NECESSARY_INTERVENTION #data is ind, dsd predicts ood, prediction is incorrect
        self.root.right.right.left.consequence = UNNECESSARY_INTERVENTION #data is ood, dsd predicts ood, prediction is correct
        self.root.right.right.right.consequence = NECESSARY_INTERVENTION #data is ood, dsd predicts ood, prediction is incorrect


        #data is ood but dsd predicts ind

        self.root.right.left.left.consequence = CORRECT_DIAGNOSIS #data is ood, dsd predicts ind, prediction is correct
        self.root.right.left.right.consequence = MISDIAGNOSIS #data is ood, dsd predicts ind, prediction is incorrect


        #data is ind, dsd predicts ind

        self.root.left.left.left.consequence = CORRECT_DIAGNOSIS  # data is ind, dsd predicts ind, prediction is correct (no intervention)
        self.root.left.left.right.consequence = MISDIAGNOSIS  # data is ind, dsd predicts ind, prediction is incorrect (loss)

    def print_tree(self):
        np.set_printoptions(precision=2)
        print("\t\t\t\tRoot\t\t\t\t")
        print(f"\t\t{self.root.left.probability:.2f} \t\t\t {self.root.right.probability:.2f}")
        print(f"\t{self.root.left.left.probability:.2f}\t\t\t {self.root.left.right.probability:.2f}\t\t\t{self.root.right.left.probability:.2f}\t\t\t{self.root.right.right.probability:.2f}")
        print(f"{self.root.left.left.left.probability:.2f}\t\t\t {self.root.left.left.right.probability:.2f}\t\t\t {self.root.left.right.left.probability:.2f}, \t\t\t {self.root.left.right.right.probability:.2f},"
              f"\t\t\t {self.root.right.left.left.probability:.2f}, \t\t\t {self.root.right.left.right.probability:.2f}, \t\t\t {self.root.right.right.left.probability:.2f}, \t\t\t {self.root.right.right.right.probability:.2f}")
        print(f"{self.root.left.left.left.consequence:.2f}\t\t\t {self.root.left.left.right.consequence:.2f}\t\t\t {self.root.left.right.left.consequence:.2f}, \t\t\t {self.root.left.right.right.consequence:.2f},"
              f"\t\t\t {self.root.right.left.left.consequence:.2f}, \t\t\t {self.root.right.left.right.consequence:.2f}, \t\t\t {self.root.right.right.left.consequence:.2f}, \t\t\t {self.root.right.right.right.consequence:.2f}")

class BaseEventTree(RiskModel):
    def __init__(self, dsd_tpr, dsd_tnr, ood_acc, ind_acc, estimator=BernoulliEstimator):
        super().__init__(estimator=estimator)
        self.ood_acc = ood_acc
        self.ind_acc = ind_acc
        self.dsd_tpr, self.dsd_tnr = dsd_tpr, dsd_tnr
        self.rate_estimator.update_tpr_tnr(dsd_tpr, dsd_tnr) #todo, dumb shortcut

        self.root = RiskNode(1)
        self.update_tree()

    def get_true_risk_for_sample(self, data):
        assert len(data) == 1
        if data["correct_prediction"].all():
            return CORRECT_DIAGNOSIS
        else:
            return MISDIAGNOSIS


    def update_tree(self):
        # print(self.rate)
        self.root.left = RiskNode(1-self.rate)
        self.root.right = RiskNode(self.rate)
        self.root.left.left = RiskNode(self.ind_acc) #data is ind, prediction is correct
        self.root.left.left.corresponds_to_correct_prediction = True
        self.root.left.right = RiskNode(1-self.ind_acc) #data is ind, prediction is incorrect
        self.root.right.left = RiskNode(self.ood_acc)
        self.root.right.left.corresponds_to_correct_prediction = True
        self.root.right.right = RiskNode(1-self.ood_acc)
        self.root.left.left.consequence = CORRECT_DIAGNOSIS
        self.root.left.right.consequence = MISDIAGNOSIS
        self.root.right.left.consequence = CORRECT_DIAGNOSIS
        self.root.right.right.consequence = MISDIAGNOSIS

    def print_tree(self):
        print("\t\t\t\tRoot\t\t\t\t")
        print(f"\t\t{self.root.left.probability} \t\t\t {self.root.right.probability}")
        print(f"\t{self.root.left.left.probability}\t\t\t {self.root.left.right.probability}\t\t\t{self.root.right.left.probability}\t\t\t{self.root.right.right.probability}")

