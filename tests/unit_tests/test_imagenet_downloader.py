# tests/test_imagenet_downloader.py

import unittest
import requests
from unittest.mock import patch, mock_open, MagicMock
from src.data_management.imagenet_downloader import download_tiny_imagenet  

class TestImagenetDownloader(unittest.TestCase):

    @patch('requests.get')
    def test_download_tiny_imagenet_successful(self, mock_get):
        """Test successful download and extraction of Tiny ImageNet."""
        # Setup mock response to mimic successful download with correct content type
        mock_response = mock_get.return_value.__enter__.return_value
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {'Content-Type': 'application/zip'}
        mock_response.iter_content.return_value = [b'x']*10  

        # Mock zipfile to test extraction without actual files
        mock_zip = MagicMock()
        with patch('builtins.open', mock_open()), \
             patch('zipfile.ZipFile', return_value=mock_zip), \
             patch('os.makedirs') as mock_makedirs:
            
            download_tiny_imagenet('http://example.com/tiny-imagenet', '/fake/path')
            
            # Assert that the zipfile was opened and extracted
            mock_zip.extractall.assert_called_once()
            # Assert that directories are created
            mock_makedirs.assert_called_once_with('/fake/path')
            # Assert the file was written to disk
            mock_zip.__enter__().write.assert_called()

    @patch('requests.get')
    def test_download_tiny_imagenet_failure(self, mock_get):
        """Test handling of non-zip content type and HTTP errors."""
        # Setup mock response to mimic an HTTP error
        mock_response = mock_get.return_value.__enter__.return_value
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("HTTP error occurred")

        with patch('builtins.print') as mock_print:
            download_tiny_imagenet('http://example.com/tiny-imagenet', '/fake/path')
            mock_print.assert_called_with("HTTP error occurred: HTTP error occurred")

        # Test for incorrect content type
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.raise_for_status = MagicMock()  

        with patch('builtins.print') as mock_print:
            download_tiny_imagenet('http://example.com/tiny-imagenet', '/fake/path')
            mock_print.assert_called_with("Failed to download: Not a zip file.")

if __name__ == '__main__':
    unittest.main()
