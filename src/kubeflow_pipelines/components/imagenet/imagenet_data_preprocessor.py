import kfp
from kfp.v2.dsl import component, InputPath, OutputPath

base_image = 'gcr.io/coms-6156-kubeflow/imagenet:latest'

@component(base_image=base_image)
def process_imagenet_data(image_dir: InputPath(), processed_dir: OutputPath()):
    from PIL import Image
    import os

    def resize_images(input_dir, output_dir, size=(224, 224)):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    img_resized = img.resize(size, Image.ANTIALIAS)
                    img_resized.save(os.path.join(output_dir, img_name))
            except IOError as e:
                print(f"Could not process image {img_name}: {e}")

    # Process each class directory in the ImageNet dataset
    for class_id in os.listdir(image_dir):
        class_dir = os.path.join(image_dir, class_id)
        output_class_dir = os.path.join(processed_dir, class_id)
        resize_images(class_dir, output_class_dir)
        print(f"Processed and resized images for class {class_id}")

