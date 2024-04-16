import shap
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Load the model
model = torch.load('model.pth')
model.eval()

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_dataset = MNIST('', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Select a subset of the test data for SHAP analysis
test_images, test_labels = next(iter(test_loader))

# Wrap the model with a function that takes a DataLoader and outputs predictions
def model_predict(data_loader):
    model.eval()
    outputs = []
    for data, _ in data_loader:
        output = model(data)
        outputs.append(output.detach().numpy())
    return outputs

# Initialize SHAP Deep Explainer
explainer = shap.DeepExplainer(model, test_images)

# Compute SHAP values
shap_values = explainer.shap_values(test_images)

# Plot SHAP values
shap.image_plot(shap_values, -test_images.numpy())