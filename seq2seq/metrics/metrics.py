"""Defines different metrics used for evaluation of tasks."""
import functools
import numpy as np
import scipy
import math
import sklearn
from logging import getLogger
from typing import Callable, Dict, List, Tuple
from .qa_utils import normalize_squad, qa_metrics

logger = getLogger(__name__)

def accuracy(predictions, targets) -> dict:
    """Computes the average accuracy."""
    return {"accuracy": 100 * ((np.array(predictions) == np.array(targets)).mean())}


def pearson_corrcoef(predictions, targets) -> dict:
    """Computes Pearson correlation coefficient."""
    from data.postprocessors import string_to_float
    targets = [string_to_float(target) for target in targets]
    predictions= [string_to_float(prediction) for prediction in predictions]
    pearson_corrcoef = 100 * scipy.stats.pearsonr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(pearson_corrcoef):
        pearson_corrcoef = 0
    return {"pearson": pearson_corrcoef}


def spearman_corrcoef(predictions, targets) -> dict:
    """Computes Spearman correlation coefficient."""
    # TODO: we need to do postprocessors in a clean way for each dataset.
    from data.postprocessors import string_to_float
    targets = [string_to_float(target) for target in targets]
    predictions= [string_to_float(prediction) for prediction in predictions]
    spearman_corrcoef = 100 * scipy.stats.spearmanr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(spearman_corrcoef):
        spearman_corrcoef = 0
    return {"spearmanr": spearman_corrcoef}


def f1_score_with_invalid(predictions, targets) -> dict:
    """Computes F1 score,  with any prediction != 0 or 1 is counted as incorrect.
    Args:
      targets: list of targets, either 0 or 1
      predictions: list of predictions, any integer value
    Returns:
      F1 score, where any prediction != 0 or 1 is counted as wrong.
    """
    def binary_reverse(labels):
       return ['0' if label == '1' else '1' for label in labels]
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    # Get indices of invalid predictions.
    invalid_idx_mask = np.logical_and(predictions != '0', predictions != '1')
    # For any prediction != 0 or 1, we set the prediction to the opposite of its corresponding target.
    predictions[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask])
    targets = targets.astype(np.int32)
    predictions = predictions.astype(np.int32)
    return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions)}


# TODO: maybe gaurd against invalid values https://stackoverflow.com/questions/56865344/how-do-i-calculate-the-matthews-correlation-coefficient-in-tensorflow
def matthews_corrcoef(predictions, targets) -> dict:
    """Computes the Matthews correlation coefficient."""
    return {"matthews_correlation": 100 * sklearn.metrics.matthews_corrcoef(targets, predictions)}

def squad(targets, predictions):
  """Computes SQuAD metrics, maximizing over answers per question.
  Args:
    targets: list of lists of strings
    predictions: list of strings
  Returns:
    dict with score_key: squad score across all targets and predictions
  """
  targets = [[normalize_squad(t) for t in u] for u in targets]
  predictions = [normalize_squad(p) for p in predictions]
  return qa_metrics(targets, predictions)




