import kfp
from kfp.v2.dsl import component, InputPath, OutputPath

base_image = 'gcr.io/coms-6156-kubeflow/imagenet:latest'

@component
def validate_imagenet_model(
    model_dir: InputPath,
    data_dir: InputPath,
    result_path: OutputPath,
    batch_size: int = 64
):
    """Validates a trained ResNet-18 model on the ImageNet validation dataset."""
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    import torchvision.models as models
    import os
    import json

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define transformations for the validation data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load ImageNet validation data
    val_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the model
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(val_dataset.classes))
    model_path = os.path.join(model_dir, 'imagenet_resnet18.pth')
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the validation images: {accuracy}%')

    results = {'accuracy': accuracy}
    with open(result_path, 'w') as result_file:
        json.dump(results, result_file)

    print(f'Results saved to {result_path}')
