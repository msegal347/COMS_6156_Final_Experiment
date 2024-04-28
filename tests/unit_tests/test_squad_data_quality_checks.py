import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.data_management.data_quality_squad import squad_data_quality_checks 

class TestDataQualitySquad(unittest.TestCase):

    @patch('pandas.read_csv')
    @patch('os.path.exists', return_value=True)
    @patch('logging.error')
    @patch('logging.warning')
    @patch('logging.info')
    def test_squad_data_quality_checks(self, mock_info, mock_warning, mock_error, mock_exists, mock_read_csv):
        """Test the data quality checks for SQuAD metadata."""
        # Setup DataFrame mock
        data = {
            'id': ['q1', 'q2', 'q3', 'q4', 'q5'],
            'context': ['This is a test.'] * 5,
            'question': ['What is this?'] * 5
        }
        df = pd.DataFrame(data)
        df.loc[4, 'context'] = ''  
        mock_read_csv.return_value = df

        # Call the function with mocked data
        squad_data_quality_checks('fake_squad_data.csv')
        
        # Check that logging captures the appropriate messages
        mock_info.assert_called_with("SQuAD data quality checks passed.")
        mock_error.assert_called_with("Empty strings found in 'context' or 'question' columns.")
        mock_exists.assert_called_once_with('fake_squad_data.csv')

        # Test missing values
        df_with_missing = df.copy()
        df_with_missing.at[0, 'context'] = None
        mock_read_csv.return_value = df_with_missing

        squad_data_quality_checks('fake_squad_data.csv')
        mock_error.assert_called_with("Missing values found in SQuAD dataset.")

        # Test non-unique ID
        df_non_unique_id = df.copy()
        df_non_unique_id.at[4, 'id'] = 'q1'
        mock_read_csv.return_value = df_non_unique_id

        squad_data_quality_checks('fake_squad_data.csv')
        mock_error.assert_called_with("'id' column in SQuAD dataset is not unique.")

if __name__ == '__main__':
    unittest.main()
