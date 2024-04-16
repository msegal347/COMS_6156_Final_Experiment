from kfp.dsl import pipeline
import kfp.compiler as compiler

@pipeline(
    name="ImageNet Data Management Pipeline",
    description="A pipeline that manages the ImageNet dataset."
)
def imagenet_data_management_pipeline(imagenet_data_path: str = '/tmp/imagenet'):
    # ImageNet Data Management
    imagenet_management = imagenet_data_management(imagenet_data_path)

# Compile the ImageNet pipeline
if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=imagenet_data_management_pipeline,
        package_path='imagenet_data_management_pipeline.yaml'
    )
