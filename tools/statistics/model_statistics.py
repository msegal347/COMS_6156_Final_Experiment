from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error, top_k_accuracy_score
from scipy.stats import ttest_rel
import numpy as np

def evaluate_classification_model(y_true, y_pred):
    """
    Evaluate a classification model using Accuracy, Precision, Recall, and F1-score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return accuracy, precision, recall, f1

def evaluate_regression_model(y_true, y_pred):
    """
    Evaluate a regression model using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae

def perform_paired_t_test(model1_results, model2_results):
    """
    Perform a paired t-test between two models' results.
    """
    t_statistic, p_value = ttest_rel(model1_results, model2_results)
    return t_statistic, p_value

def evaluate_top_k_accuracy(y_true, y_scores, k=1):
    """
    Calculate Top-k accuracy for a given set of true labels and predicted scores.
    """
    # Ensure k does not exceed the number of classes
    max_k = min(k, y_scores.shape[1])
    if max_k != k:
        print(f"Adjusted k from {k} to {max_k} due to class count constraints.")
    top_k_accuracy = top_k_accuracy_score(y_true, y_scores, k=max_k)
    return top_k_accuracy