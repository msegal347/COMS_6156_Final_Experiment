import unittest
from unittest.mock import patch, mock_open
from src.data_management.squad_downloader import download_file, validate_file_size

class TestSquadDownloader(unittest.TestCase):

    @patch('requests.get')
    def test_download_file(self, mock_get):
        """Test that the file download function behaves correctly when making requests."""
        mock_response = mock_get.return_value
        mock_response.iter_content.return_value = [b'data']*10
        mock_response.status_code = 200

        with patch('builtins.open', mock_open()) as mocked_file:
            download_file('http://example.com/file', '/fake/path/to/file')
            mocked_file.assert_called_once_with('/fake/path/to/file', 'wb')
            self.assertEqual(mocked_file().write.call_count, 10)
            mock_get.assert_called_once_with('http://example.com/file', stream=True)

    @patch('os.path.getsize')
    def test_validate_file_size(self, mock_getsize):
        """Test the file size validation logic under both passing and failing conditions."""
        mock_getsize.return_value = 2048  # 2 KB
        with patch('builtins.print') as mock_print:
            validate_file_size('/fake/path/to/file', 1)  #
            mock_print.assert_called_with('Validation passed for /fake/path/to/file')
            
            validate_file_size('/fake/path/to/file', 3000)  
            mock_print.assert_called_with('Validation failed for /fake/path/to/file, size: 2.0 KB')

if __name__ == '__main__':
    unittest.main()
