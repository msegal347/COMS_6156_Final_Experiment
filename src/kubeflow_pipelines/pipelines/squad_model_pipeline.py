import kfp
from kfp import dsl
import kfp.components as comp

# Load components from YAML files
train_op = comp.load_component_from_file('path/to/train_component.yaml')
validate_op = comp.load_component_from_file('path/to/validate_component.yaml')
interpret_op = comp.load_component_from_file('path/to/interpret_component.yaml')

@dsl.pipeline(
    name='SQuAD Model Training, Validation, and Interpretation Pipeline',
    description='A pipeline for training, validating, and interpreting a BERT model on the SQuAD dataset.'
)
def squad_model_pipeline(
        dataset_path: str,
        model_save_path: str,
        validation_output_path: str,
        interpretation_output_path: str):
    
    # Training step
    train_task = train_op(
        dataset_path=dataset_path,
        model_save_path=model_save_path
    )
    
    # Validation step, assuming it needs the model path and dataset path
    validate_task = validate_op(
        model_path=train_task.outputs['model_save_path'],
        dataset_path=dataset_path,
        output_path=validation_output_path
    )

    # Interpretation step, assuming it requires the model path
    interpret_task = interpret_op(
        model_path=train_task.outputs['model_save_path'],
        dataset_path=dataset_path,
        output_path=interpretation_output_path
    )
