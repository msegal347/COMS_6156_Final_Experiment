import unittest
from unittest.mock import patch, MagicMock, mock_open
from PIL import Image
from src.data_management.data_preprocessor_imagenet import resize_images, process_imagenet 

class TestDataPreprocessorImagenet(unittest.TestCase):

    @patch('os.makedirs')
    @patch('os.listdir', return_value=['image1.jpg', 'image2.jpg'])
    @patch('PIL.Image.open')
    @patch('os.path.exists', return_value=True)
    def test_resize_images(self, mock_exists, mock_open, mock_listdir, mock_makedirs):
        """Test the resizing of images within a directory."""
        mock_image = MagicMock(spec=Image.Image)
        mock_open.return_value = mock_image
        mock_image.resize.return_value = mock_image

        resize_images('fake/image_dir', 'fake/output_dir')

        # Check that makedirs was called to ensure output directory exists
        mock_makedirs.assert_called_with('fake/output_dir')
        
        # Check that images are opened and resized
        self.assertEqual(mock_open.call_count, 2)
        mock_image.resize.assert_called_with((224, 224), Image.ANTIALIAS)
        mock_image.save.assert_called()

    @patch('data_preprocessor_imagenet.resize_images')
    @patch('os.listdir', return_value=['class1', 'class2'])
    @patch('os.makedirs')
    def test_process_imagenet(self, mock_makedirs, mock_listdir, mock_resize_images):
        """Test the processing of ImageNet directory."""
        process_imagenet('fake/raw_dir', 'fake/processed_dir')

        # Check that resize_images is called for each class directory
        self.assertEqual(mock_resize_images.call_count, 2)
        calls = [
            (('fake/raw_dir/class1', 'fake/processed_dir/class1'),),
            (('fake/raw_dir/class2', 'fake/processed_dir/class2'),)
        ]
        mock_resize_images.assert_has_calls(calls, any_order=True)

if __name__ == '__main__':
    unittest.main()
