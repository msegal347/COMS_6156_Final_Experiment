from sklearn.metrics import accuracy_score, top_k_accuracy_score
import numpy as np

def evaluate_imagenet_classification(y_true, y_pred, y_scores):
    """
    Evaluate ImageNet classification model performance.
    Args:
        y_true (np.array): True class indices for each example.
        y_pred (np.array): Predicted class indices for each example.
        y_scores (np.array): Score/probability for each class for each example.

    Returns:
        dict: Dictionary containing accuracy, top-1 and top-5 accuracy.
    """
    accuracy = accuracy_score(y_true, y_pred)
    top1_accuracy = top_k_accuracy_score(y_true, y_scores, k=1)
    top5_accuracy = top_k_accuracy_score(y_true, y_scores, k=5)

    return {
        "accuracy": accuracy,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy
    }

