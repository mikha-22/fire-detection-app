# src/ml_model/fire_detection_model.py

import os # Added os import for path joining
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class DummyFireDetectionModel(nn.Module):
    """
    A dummy PyTorch model for wildfire detection.
    For MVP, it simply acts as a placeholder and always returns a predefined "detected" state.
    In a real scenario, this would be a trained Convolutional Neural Network (CNN).
    """
    def __init__(self):
        super(DummyFireDetectionModel, self).__init__()
        # For a real model, you'd define your layers here, e.g.:
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
        # self.pool = nn.MaxPool2d(2)
        # self.fc = nn.Linear(..., 2) # e.g., for binary classification (fire/no-fire)

        # We'll use a placeholder parameter so TorchServe finds something to load
        self.dummy_param = nn.Parameter(torch.randn(1))

    def forward(self, x):
        """
        Forward pass for the dummy model.
        In a real model, x would be an image tensor, and you'd pass it through your layers.
        For this dummy, we ignore the input and return a fixed output.
        """
        # For MVP, simulate a fixed output: assume fire detected with 85% confidence
        # Real model would output logits or probabilities, e.g., torch.tensor([[0.15, 0.85]])
        return torch.tensor([[0.15, 0.85]]) # [no_fire_prob, fire_prob]

# Define a transformation pipeline that mimics what a real model would need for input.
# This will be used in the handler's preprocess method.
# Assuming input images are 224x224 RGB for a typical CNN.
MODEL_INPUT_TRANSFORMS = Compose([
    Resize((224, 224)),       # Resize to expected input size
    ToTensor(),               # Convert PIL Image to PyTorch Tensor (HWC to CHW, 0-255 to 0-1)
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization (common)
])

if __name__ == '__main__':
    # Simple test of the dummy model
    model = DummyFireDetectionModel()
    dummy_input = torch.randn(1, 3, 224, 224) # Batch size 1, 3 channels, 224x224 image
    output = model(dummy_input)
    print("Dummy Model Output (logits/probabilities):", output)
    # Convert logits/probabilities to predicted class (0 for no fire, 1 for fire)
    predicted_class = torch.argmax(output, dim=1).item()
    confidence = output.softmax(dim=1)[0][predicted_class].item()
    print(f"Predicted Class: {'Fire' if predicted_class == 1 else 'No Fire'} (Confidence: {confidence:.2f})")

    # --- MODIFIED SAVE LOGIC ---
    # Save a dummy state_dict for packaging, named 'model.pth'
    # and place it directly within the 'src/ml_model/' directory.
    model_filename = "model.pth"
    # Get the directory where this script (fire_detection_model.py) is located
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_script_directory, model_filename)
    
    torch.save(model.state_dict(), save_path)
    print(f"Dummy model state_dict saved to {save_path}")
    # --- END MODIFIED SAVE LOGIC ---
