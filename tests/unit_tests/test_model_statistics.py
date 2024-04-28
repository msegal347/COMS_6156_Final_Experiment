import unittest
from unittest.mock import patch
from tools.statistics.model_statistics import evaluate_classification_model, evaluate_regression_model, perform_paired_t_test
import numpy as np

class TestModelPerformanceAnalysis(unittest.TestCase):
    def test_evaluate_classification_model(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        acc, prec, rec, f1 = evaluate_classification_model(y_true, y_pred)
        self.assertAlmostEqual(acc, 0.75)
        self.assertAlmostEqual(prec, 1.0)
        self.assertAlmostEqual(rec, 0.5)
        self.assertAlmostEqual(f1, 0.666, places=3)

    def test_evaluate_regression_model(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        mse, mae = evaluate_regression_model(y_true, y_pred)
        self.assertAlmostEqual(mse, 0.01)
        self.assertAlmostEqual(mae, 0.1)

    def test_perform_paired_t_test(self):
        results1 = np.array([1, 2, 3])
        results2 = np.array([1, 2, 3])
        t_stat, p_val = perform_paired_t_test(results1, results2)
        self.assertAlmostEqual(t_stat, 0.0)
        self.assertTrue(np.isnan(p_val))  

if __name__ == '__main__':
    unittest.main()
