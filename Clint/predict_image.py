import torch
from torchvision import models
from torchvision.models import ViT_B_16_Weights
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Define class names (in the same order used during training)
class_names = ['doublemint', 'mouse', 'suns_cream']

# Load the model with the saved weights
def load_model(weights_path, num_classes, device):
    model = models.vit_b_16(weights=None)  # We don't load pretrained weights here
    model.heads = torch.nn.Linear(in_features=768, out_features=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Define the exact transforms used during training
pretrained_vit_weights = ViT_B_16_Weights.DEFAULT
transform = pretrained_vit_weights.transforms()

# Predict function for a single image
def predict_image(image_path, model, transform, class_names, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_idx = outputs.argmax(1).item()
    return class_names[predicted_idx]

# Example usage
if __name__ == "__main__":
    weights_path = "vit_model_weights.pth"
    image_path = r"image6.jpg"

    model = load_model(weights_path, num_classes=len(class_names), device=device)
    prediction = predict_image(image_path, model, transform, class_names, device)

    # print(f"✅ Predicted class: {prediction}")
    print(prediction)
