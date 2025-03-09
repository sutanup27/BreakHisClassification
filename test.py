import torch
from VGG import VGG  # Ensure you import your correct model architecture

# Initialize the model
model = VGG()  # Replace with your actual model class

# Load the saved state_dict correctly
state_dict = torch.load("vgg.16.pth", map_location=torch.device('cpu'))  # Use 'cpu' if necessary
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

# Print out missing/unexpected keys for debugging
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

# Set model to evaluation mode
model.eval()
input_tensor=torch.randn(1, 3, 224, 224)
# Now you can run predictions
output = model(input_tensor)  # Ensure input_tensor is properly formatted
