from PIL import Image
import os

def resize_images(image_dir, output_dir, size=(224, 224)):
    """
    Resizes all images in the specified directory to the specified size.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path)
        img_resized = img.resize(size, Image.ANTIALIAS)
        img_resized.save(os.path.join(output_dir, img_name))

def process_imagenet(raw_dir, processed_dir):
    """
    Processes the ImageNet dataset by resizing all images to 224x224 pixels.
    """
    for class_id in os.listdir(raw_dir):
        class_dir = os.path.join(raw_dir, class_id)
        output_class_dir = os.path.join(processed_dir, class_id)
        resize_images(class_dir, output_class_dir)
        print(f"Processed and resized images for class {class_id}")

if __name__ == "__main__":
    process_imagenet('./data/raw/tiny_imagenet', './data/processed/tiny_imagenet')
