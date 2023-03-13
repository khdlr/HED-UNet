import torch
from collections import defaultdict
from .utils import sample_map
import numpy as np

class_names = ['NA', 'Rock', 'Glacier', 'Ocean']

@torch.no_grad()
def compute_premetrics(y_pred, y_true):
  y_pred = y_pred.exp() > 0.5

  return dict(
    TP = ((y_pred == 1) & (y_true == 1)).long().sum(),
    TN = ((y_pred == 0) & (y_true == 0)).long().sum(),
    FP = ((y_pred == 1) & (y_true == 0)).long().sum(),
    FN = ((y_pred == 0) & (y_true == 1)).long().sum(),
  )


class Metrics():
    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = {}
        self.running_agg = defaultdict(float)
        self.running_count = defaultdict(int)
        self.running_confusion_matrix = np.zeros([4, 4], dtype=np.int64)
        self.running_confusion_matrix_mask = np.zeros([2, 2], dtype=np.int64)

    @torch.no_grad()
    def step(self, terms):
      for term in terms:
        self.running_agg[term] += terms[term].cpu().numpy()
        self.running_count[term] += 1

    def update_terms(self, y_pred, y_true, kind):
      if kind in ['Termpicks', 'Kochtitzky', 'Fronts', 'TUD_Fronts']:
        self._update(kind, compute_premetrics(y_pred, y_true))
      elif kind == 'Mask':
        y_pred = y_pred.argmax(dim=0)
        confusion_idx = y_true.flatten() + 2 * y_pred.flatten()
        confusion_matrix = torch.bincount(confusion_idx, minlength=(2*2))
        confusion_matrix = confusion_matrix.cpu().numpy().reshape(2, 2)
        self.running_confusion_matrix_mask += confusion_matrix
      elif kind == 'Zones':
        y_pred = y_pred.argmax(dim=0)
        confusion_idx = y_true.flatten() + 4 * y_pred.flatten()
        confusion_matrix = torch.bincount(confusion_idx, minlength=(4*4))
        confusion_matrix = confusion_matrix.cpu().numpy().reshape(4, 4)
        self.running_confusion_matrix += confusion_matrix

    def _update(self, kind, values):
      if kind not in self.counts:
        self.counts[kind] = defaultdict(int)
      for k in values:
        self.counts[kind][k] += values[k]

    @torch.no_grad()
    def peek(self):
        values = {}
        for key in self.running_agg:
            values[key] = float(self.running_agg[key] / self.running_count[key])
        for kind in self.counts:
          TP = self.counts[kind]['TP']
          TN = self.counts[kind]['TN']
          FP = self.counts[kind]['FP']
          FN = self.counts[kind]['FN']

          values[f'{kind}/Accuracy'] = (TP + TN) / (TP + TN + FP + FN)
          values[f'{kind}/IoU'] = (TP) / (TP + FP + FN)
          values[f'{kind}/mIoU'] = 0.5 * (TP / (TP + FP + FN) + TN / (TN + FP + FN))
          values[f'{kind}/Recall'] = (TP) / (TP + FN),
          values[f'{kind}/Precision'] = (TP) / (TP + FP)
          values[f'{kind}/F1'] = (2 * TP) / (2 * TP + FP + FN)

        ## Masks
        CM = self.running_confusion_matrix
        TP = np.diag(CM)  # diagonal: prediction == ground_truth
        FP = CM.sum(axis=1) - TP  # FP = #PixelsActuallyInClass - TP
        FN = CM.sum(axis=0) - TP  # FN = #PixelsPredictedAsClass - TP
        TN = CM.sum() - TP - FP - FN  # FN = #Pixels - TP - FP - FN
        IoU = TP / (FP + FN + TP)
        values[f'Mask/mIoU'] = IoU.mean()
        for name, value in zip(['Ocean', 'Land'], IoU):
          values[f'Mask/IoU_{name}'] = value

        ## Multiclass
        CM = self.running_confusion_matrix
        TP = np.diag(CM)  # diagonal: prediction == ground_truth
        FP = CM.sum(axis=1) - TP  # FP = #PixelsActuallyInClass - TP
        FN = CM.sum(axis=0) - TP  # FN = #PixelsPredictedAsClass - TP
        TN = CM.sum() - TP - FP - FN  # FN = #Pixels - TP - FP - FN
        IoU = TP / (FP + FN + TP)
        values[f'Zones/mIoU'] = IoU.mean()
        for name, value in zip(class_names, IoU):
          values[f'Zones/IoU_{name}'] = value

        return values


    @torch.no_grad()
    def evaluate(self):
        values = self.peek()
        self.reset()
        return values
