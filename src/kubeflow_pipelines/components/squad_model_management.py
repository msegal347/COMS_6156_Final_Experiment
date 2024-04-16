from kfp import components

train_op = components.load_component_from_image(
    'your-registry/squad-training:latest'
)

validate_op = components.load_component_from_image(
    'your-registry/squad-validation:latest'
)

interpret_op = components.load_component_from_image(
    'your-registry/squad-interpretation:latest'
)
