import numpy as np


def true_positive(prediction, target):
    return ((prediction > 0) * (target > 0.5)).float().sum()


def false_positive(prediction, target):
    return ((prediction > 0) * (target <= 0.5)).float().sum()


def false_negative(prediction, target):
    return ((prediction <= 0) * (target > 0.5)).float().sum()


def true_negative(prediction, target):
    return ((prediction <= 0) * (target <= 0.5)).float().sum()


AGGREGATORS = {
    'TP': true_positive,
    'FP': false_positive,
    'FN': false_negative,
    'TN': true_negative,
}


class Metrics():
    def __init__(self, *metrics):
        self.metrics = metrics
        self.required_aggregators = set()
        for m in self.metrics:
            self.required_aggregators |= m.required_aggregators()
        self.reset()

    def reset(self):
        self.state = {agg: 0 for agg in self.required_aggregators}
        self.running_agg = {}
        self.running_count = {}

    def step(self, prediction, target, **additional_terms):
        for agg in self.required_aggregators:
            self.state[agg] += AGGREGATORS[agg](prediction, target)

        for term in additional_terms:
            agg = self.running_agg.get(term, 0) + additional_terms[term]
            count = self.running_count.get(term, 0) + 1

            self.running_agg[term] = agg
            self.running_count[term] = count

    def evaluate(self):
        values = {}
        for m in self.metrics:
            values[m.__name__] = m.evaluate(self.state)
        for key in self.running_agg:
            values[key] = self.running_agg[key] / self.running_count[key]
        self.reset()
        return values


class Accuracy():
    @staticmethod
    def required_aggregators():
        return set(['TP', 'FP', 'FN', 'TN'])

    @staticmethod
    def evaluate(state):
        correct = state['TP'] + state['TN']
        wrong = state['FP'] + state['FN']
        if correct + wrong == 0:
            return np.nan
        return correct / (correct + wrong)


class Precision():
    @staticmethod
    def required_aggregators():
        return set(['TP', 'FP'])

    @staticmethod
    def evaluate(state):
        if state['TP'] + state['FP'] == 0:
            return np.nan
        return state['TP'] / (state['TP'] + state['FP'])


class Recall():
    @staticmethod
    def required_aggregators():
        return set(['TP', 'FN'])

    @staticmethod
    def evaluate(state):
        if state['TP'] + state['FN'] == 0:
            return np.nan
        return state['TP'] / (state['TP'] + state['FN'])


class F1():
    @staticmethod
    def required_aggregators():
        return set(['TP', 'FP', 'FN', 'TN'])

    @staticmethod
    def evaluate(state):
        precision = Precision.evaluate(state)
        recall = Recall.evaluate(state)
        if precision + recall == 0:
            return np.nan
        return 2 * precision * recall / (precision + recall)


class IoUWater():
    @staticmethod
    def required_aggregators():
        return set(['TN', 'FP', 'FN'])

    @staticmethod
    def evaluate(state):
        return state['TN'] / (state['TN'] + state['FP'] + state['FN'])


class IoULand():
    @staticmethod
    def required_aggregators():
        return set(['TP', 'FP', 'FN'])

    @staticmethod
    def evaluate(state):
        return state['TP'] / (state['TP'] + state['FP'] + state['FN'])


class mIoU():
    @staticmethod
    def required_aggregators():
        return set(['TP', 'FP', 'FN', 'TN'])

    @staticmethod
    def evaluate(state):
        iou_land = IoULand.evaluate(state)
        iou_water = IoUWater.evaluate(state)
        return 0.5 * (iou_land + iou_water)
