import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import shap
import numpy as np
import matplotlib.pyplot as plt

def load_imagenet_data(batch_size=1, num_batches=1):
    """Load a subset of ImageNet data for demonstration."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    imagenet_data_path = "./data/imagenet/processed"
    imagenet_dataset = datasets.ImageFolder(root=f"{imagenet_data_path}/val", transform=transform)
    imagenet_loader = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True)
    
    return imagenet_loader

def interpret_model_with_shap(model, data_loader):
    """Perform model interpretability using SHAP."""
    # Wrap the model with a wrapper that returns the output logits
    def model_wrapper(x):
        with torch.no_grad():
            return model(x)

    # Load a batch of images (for demonstration, we'll explain just one batch)
    images, _ = next(iter(data_loader))
    
    # Define a background dataset (a subset of images) to compute SHAP values
    background = images[:1]
    test_images = images[1:2]
    
    # Initialize SHAP Deep Explainer
    explainer = shap.DeepExplainer(model_wrapper, background)
    shap_values = explainer.shap_values(test_images)
    
    # Plot SHAP values
    shap.image_plot(shap_values, -test_images.numpy())

if __name__ == "__main__":
    # Load a pretrained ResNet18 model
    model_path = './models/saved_models/imagenet_resnet18.pth' 
    model = resnet18(pretrained=False)
    #model = resnet18(pretrained=True)
    model.eval()

    # Load ImageNet data
    imagenet_loader = load_imagenet_data()

    # Interpret the model with SHAP
    interpret_model_with_shap(model, imagenet_loader)
