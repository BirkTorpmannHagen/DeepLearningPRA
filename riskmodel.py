import numpy as np

from rateestimators import BernoulliEstimator

MISDIAGNOSIS = 6100
UNNECESSARY_INTERVENTION = 635+100+0.2*MISDIAGNOSIS
NECESSARY_INTERVENTION = 635+100+0.2*MISDIAGNOSIS #arbitrary, but lower than unnecessary intervention
CORRECT_DIAGNOSIS = 635 # cost of correct diagnosis during AI screening


class RiskNode:
    def __init__(self, probability, consequence=0, left=None, right=None):
        self.left = None
        self.right = None
        self.probability = probability #probability of right child
        self.consequence = 0 #if non leaf node

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
    def __init__(self, dsd_tpr, dsd_tnr, ind_ndsd_acc, maximum_loss, estimator):
        """
                Binary Tree defined by the following structure:
                ood+/- -> dsd+/- -> prediction+/- -> consequence
                """
        super().__init__(estimator)
        self.dsd_tpr, self.dsd_tnr = dsd_tpr, dsd_tnr
        self.ind_ndsd_acc  = ind_ndsd_acc
        print(f"Initializing Risk Model with {dsd_tpr, dsd_tnr, ind_ndsd_acc}")
        self.root = RiskNode(1) #root
        self.maximum_loss = maximum_loss
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
                return CORRECT_DIAGNOSIS if loss < self.maximum_loss else MISDIAGNOSIS

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
    def __init__(self, ood_acc, ind_acc, maximum_loss=0.5, estimator=BernoulliEstimator):
        super().__init__(estimator=estimator)
        self.maximum_loss = maximum_loss
        print(self.maximum_loss)
        self.ood_acc = ood_acc
        self.ind_acc = ind_acc
        self.root = RiskNode(1)
        self.update_tree()

    def get_true_risk_for_sample(self, is_ood, loss):
        # print(loss, self.maximum_loss)
        # input()
        if loss>self.maximum_loss:
            return MISDIAGNOSIS
        else:
            return CORRECT_DIAGNOSIS
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
