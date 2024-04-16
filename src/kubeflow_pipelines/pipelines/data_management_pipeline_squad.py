from kfp.dsl import pipeline
import kfp.compiler as compiler

@pipeline(
    name="SQuAD Data Management Pipeline",
    description="A pipeline that manages the SQuAD dataset."
)
def squad_data_management_pipeline(squad_data_path: str = '/tmp/squad'):
    # SQuAD Data Management
    squad_management = squad_data_management(squad_data_path)

# Compile the SQuAD pipeline
if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=squad_data_management_pipeline,
        package_path='squad_data_management_pipeline.yaml'
    )
