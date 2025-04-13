import cv2
import torch
from torchvision import models, transforms
from torchvision.models import ViT_B_16_Weights
from PIL import Image
import warnings
import requests

warnings.filterwarnings("ignore", category=UserWarning)

# Class names used during training
class_names = ['comb', 'doublemint', 'mouse', 'no_item', 'suns_cream', 'tooth_brush']

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained ViT model architecture and adjust for 5 classes
model = models.vit_b_16(weights=None).to(device)
model.heads = torch.nn.Linear(in_features=768, out_features=len(class_names)).to(device)

# Load your trained weights
model.load_state_dict(torch.load("vit_model_weights.pth", map_location=device))
model.eval()

# Use the transform associated with the ViT pretrained weights
weights = ViT_B_16_Weights.DEFAULT
transform = weights.transforms()

# Function to predict a single frame
def predict_frame(frame, model, transform, class_names, device):
    try:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            pred_idx = predicted.item()

        if 0 <= pred_idx < len(class_names):
            return class_names[pred_idx]
        else:
            return "no_item"

    except Exception:
        return "no_item"

# Path to your recorded video
video_path=""
user_id=""
def set_video_path(path):
    global video_path
    video_path = path

def set_user_id(id):
    global user_id
    user_id=id

# Load the video using OpenCV

stack = []
def get_all_items():
    global stack
    return stack


url = "http://127.0.0.1:5000/push_data"  # Or your actual deployed URL
item_price_map = {
    "comb":30,
    "doublemint": 50,
    "suns_cream": 540
}

def custom_push(element):
    if stack and stack[-1] == element:
        return

    stack.append(element)

    if element == "no_item":
        return

    payload = {
        "id": user_id,
        "item": element,
        "price": item_price_map[element]
    }
    requests.post(url, json=payload)


def video_streaming():
    if not user_id or not video_path or not cv2.VideoCapture(video_path).isOpened():
        print("❌ Invalid or empty video path or user is not defined. Exiting function.")
        return

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Predict class for the current frame
        pred_class = predict_frame(frame, model, transform, class_names, device)

        custom_push(pred_class)

        # Overlay prediction on the frame
        cv2.putText(frame, f"Prediction: {pred_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("ViT Prediction on Video", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    video_streaming()
    print(stack)

