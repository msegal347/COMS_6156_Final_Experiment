import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
import os

def train_imagenet_model(data_dir, save_model_dir):
    """
    Trains a ResNet-18 model on the ImageNet dataset and saves the trained model.

    Parameters:
    - data_dir (str): Directory path to the ImageNet dataset.
    - save_model_dir (str): Directory path where the trained model will be saved.

    Returns:
    - None
    """
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load ImageNet training data
    train_dataset = ImageFolder(root=f'{data_dir}/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Initialize the model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # Modify the last layer for the number of classes in the dataset
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    model.train()
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    print('Finished Training ImageNet Model')

    # Save the trained model
    model_save_path = os.path.join(save_model_dir, 'imagenet_resnet18.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == "__main__":
    data_dir = '/path/to/your/imagenet/data'  # Update this path to your ImageNet data directory
    save_model_dir = './models/saved_models'  # Define the directory to save your model
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    train_imagenet_model(data_dir, save_model_dir)
