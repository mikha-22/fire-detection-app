# src/ml_model/fire_detection_model.py

import os
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image # Added for the local test block

class DummyFireDetectionModel(nn.Module):
    """
    An improved dummy PyTorch model for wildfire detection.
    Its output is controlled by the content of the input image, allowing for
    flexible testing of different scenarios (e.g., fire, no-fire).
    """
    def __init__(self):
        super(DummyFireDetectionModel, self).__init__()
        # Placeholder parameter so the model has a state_dict to save/load
        self.dummy_param = nn.Parameter(torch.randn(1))

    def forward(self, batch_tensor):
        """
        Forward pass for the dummy model. It inspects each image in the batch
        to decide on the output.

        - A predominantly RED image triggers a 'fire' detection.
        - A predominantly GREEN image triggers a 'no fire' detection.
        - Other images default to a 'no fire' detection.
        """
        outputs = []
        # The input is a batch of tensors (N, C, H, W) where C=3 (R,G,B)
        for image_tensor in batch_tensor:
            # Get the mean of each channel (Red, Green, Blue) across the image.
            # After ImageNet normalization, the original color dominance is preserved
            # in a relative sense, though the absolute values change.
            # These thresholds are empirically chosen for this normalized space.
            red_channel_mean = image_tensor[0, :, :].mean()
            green_channel_mean = image_tensor[1, :, :].mean()

            # A pure red (255,0,0) image, after normalization, will have a high value here.
            if red_channel_mean > 0.5:
                # High confidence fire: [no_fire_prob, fire_prob]
                outputs.append(torch.tensor([0.10, 0.90]))
            # A pure green (0,255,0) image will have a high value here.
            elif green_channel_mean > 0.5:
                # High confidence no-fire: [no_fire_prob, fire_prob]
                outputs.append(torch.tensor([0.95, 0.05]))
            else:
                # Default case for any other image (e.g., blue, or a real satellite image)
                outputs.append(torch.tensor([0.80, 0.20])) # Default to no-fire

        # Stack the individual tensor results into a single batch tensor for output
        return torch.stack(outputs)

# Transformation pipeline remains the same
MODEL_INPUT_TRANSFORMS = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    # --- This block now serves two purposes ---
    # 1. Save the model's state_dict for packaging.
    # 2. Create the special "trigger" images needed for testing.

    print("--- Running DummyFireDetectionModel main block ---")

    # --- Create Trigger Images for Testing ---
    # These images, when used as input, will control the model's output.
    if not os.path.exists("test_trigger_images"):
        os.makedirs("test_trigger_images")
    
    Image.new('RGB', (224, 224), color = 'red').save("test_trigger_images/force_fire_detection.png")
    Image.new('RGB', (224, 224), color = 'green').save("test_trigger_images/force_no_fire_detection.png")
    print("Created trigger images in 'test_trigger_images/' directory.")
    print("ACTION: Upload these images to your GCS bucket (e.g., gs://fire-app-bucket/mock_trigger_images/).")

    # --- Save the Model's state_dict ---
    model = DummyFireDetectionModel()
    model_filename = "model.pth"
    # Save it directly in the ml_model directory, as expected by archive_model.sh
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_script_directory, model_filename)
    
    torch.save(model.state_dict(), save_path)
    print(f"Dummy model state_dict saved to {save_path}")
    print("--- Main block finished ---")
