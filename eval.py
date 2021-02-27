import ipdb
from prettytable import PrettyTable

class Eval:
    def __init__(self, class_num):
        self.class_num = class_num
        # micro
        self.pred = [0] * class_num
        self.truth = [0] * class_num
        self.TP = [0] * class_num
        self.precision = [0] * class_num
        self.recall = [0] * class_num
        self.F1 = [0] * class_num

    def evaluate(self, preds, truths):
        for pred, truth in zip(preds, truths):
            self.pred[pred] += 1
            self.truth[truth] += 1
            if pred == truth:
                self.TP[truth] += 1

    def generate_result(self):
        for idx in range(self.class_num):
            precision = self.TP[idx] / self.pred[idx] if self.pred[idx] != 0 else 0
            recall = self.TP[idx] / self.truth[idx] if self.truth[idx] != 0 else 0
            self.precision[idx] = precision
            self.recall[idx] = recall
            self.F1[idx] = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        self.precision_macro = 0
        self.recall_macro = 0
        self.F1_macro = 0
        truth_not_zero = 0
        for idx in range(self.class_num):
            if self.truth[idx] != 0:
                self.precision_macro += self.precision[idx]
                self.recall_macro += self.recall[idx]
                self.F1_macro += self.F1[idx]
                truth_not_zero += 1
        self.precision_macro /= truth_not_zero
        self.recall_macro /= truth_not_zero
        self.F1_macro /= truth_not_zero

        total_pred = sum(self.pred)
        total_truth = sum(self.truth)
        total_TP = sum(self.TP)
        self.precision_micro = total_TP / total_pred if total_pred != 0 else 0
        self.recall_micro = total_TP / total_truth if total_truth != 0 else 0
        self.F1_micro = 2 * self.precision_micro * self.recall_micro / (self.precision_micro + self.recall_micro) if self.precision_micro + self.recall_micro != 0 else 0

    def display_result(self):
        pass