import unittest
import os
from unittest.mock import patch, mock_open
import json
from src.data_management.data_preprocessor_squad import process_squad  

class TestDataPreprocessorSquad(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
        "data": [
            {
                "paragraphs": [
                    {
                        "context": "Test context",
                        "qas": [
                            {
                                "id": "1",
                                "question": "What is testing?",
                                "answers": [
                                    {"text": "Testing is process of verifying."}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }))
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_process_squad(self, mock_to_csv, mock_makedirs, mock_file):
        # Call the function with the mocked file path and output path
        process_squad('fake_path.json', 'fake_output.csv')
        
        # Check that the open was called correctly
        mock_file.assert_called_with('fake_path.json', 'r')
        
        # Check that the directory for output was potentially created
        mock_makedirs.assert_called_with(os.path.dirname('fake_output.csv'), exist_ok=True)
        
        # Check that the data was attempted to be written to a CSV
        mock_to_csv.assert_called_once()
        args, kwargs = mock_to_csv.call_args
        self.assertEqual(args[0], 'fake_output.csv')
        self.assertEqual(kwargs['index'], False)

        print(mock_to_csv.call_args_list)

if __name__ == '__main__':
    unittest.main()
