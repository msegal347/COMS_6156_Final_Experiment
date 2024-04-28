import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.data_management.data_quality_imagenet import imagenet_data_quality_checks 

class TestDataQualityImagenet(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_imagenet_data_quality_checks(self, mock_read_csv):
        """Test the data quality checks for ImageNet metadata."""
        # Setup DataFrame mock
        data = {
            'class_id': ['class1'] * 600 + ['class2'] * 400 + ['class3'] * 100
        }
        df = pd.DataFrame(data)
        mock_read_csv.return_value = df

        # Test successful data quality check
        try:
            imagenet_data_quality_checks('fake_metadata.csv')
        except Exception as e:
            self.fail(f"Data quality checks should have passed but failed with exception {e}")

        # Test failure due to missing values
        df_with_missing = df.copy()
        df_with_missing.at[0, 'class_id'] = None
        mock_read_csv.return_value = df_with_missing

        with self.assertRaises(ValueError) as context:
            imagenet_data_quality_checks('fake_metadata.csv')
        self.assertIn("Missing values found in ImageNet metadata", str(context.exception))

        # Test failure due to insufficient images per class
        df_insufficient_images = pd.DataFrame({'class_id': ['class1'] * 600 + ['class2'] * 400 + ['class3'] * 450})
        mock_read_csv.return_value = df_insufficient_images

        with self.assertRaises(ValueError) as context:
            imagenet_data_quality_checks('fake_metadata.csv')
        self.assertIn("Some classes have fewer than 500 images", str(context.exception))

if __name__ == '__main__':
    unittest.main()
