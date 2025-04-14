import time
import warnings
import requests
import firebase_admin
import torch
import cv2
from torchvision import models
from torchvision.models import ViT_B_16_Weights
from PIL import Image
from firebase_admin import credentials, db

warnings.filterwarnings("ignore", category=UserWarning)

class AutoCheckoutSystem:
    class_names = ['comb', 'doublemint', 'mouse', 'no_item', 'suns_cream', 'tooth_brush']
    item_price_map = {
        "comb": 30,
        "doublemint": 50,
        "suns_cream": 540,
        "tooth_brush": 30,
        "mouse": 450
    }

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.vit_b_16(weights=None).to(self.device)
        self.model.heads = torch.nn.Linear(in_features=768, out_features=len(self.class_names)).to(self.device)
        self.model.load_state_dict(torch.load("vit_model_weights.pth", map_location=self.device))
        self.model.eval()
        self.weights = ViT_B_16_Weights.DEFAULT
        self.transform = self.weights.transforms()

        self.stack = []
        self.video_path = ""
        self.user_id = ""
        self.url = ""
        self.time_out_threshold_value = 600

        cred = credentials.Certificate('autocheckouts_firebase_credential.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://sdpautocheckouts-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })

    def set_video_path(self, path):
        self.video_path = path

    def set_user_id(self, id):
        self.user_id = id

    def get_all_items(self):
        return self.stack

    def set_end_point_url(self, url_):
        self.url = url_

    def set_time_out_threshold(self, time_out):
        self.time_out_threshold_value = float(time_out)

    def predict_frame(self, frame):
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(image)
                _, predicted = torch.max(outputs, 1)
                pred_idx = predicted.item()

            if 0 <= pred_idx < len(self.class_names):
                return self.class_names[pred_idx]
            else:
                return "no_item"

        except Exception:
            return "no_item"

    def custom_push(self, element):
        if self.stack and self.stack[-1] == element:
            return True

        self.stack.append(element)
        if element == "no_item":
            return True

        payload = {
            "id": self.user_id,
            "item": element,
            "price": self.item_price_map[element]
        }
        response = requests.post(self.url, json=payload)
        print(response.json()['message'])
        return response.json()['status']

    def wait_until_cart_deleted(self):
        start_time = time.time()
        cart_ref = db.reference(f'/cart/{self.user_id}')
        print('⏰ Waiting To check out the items before time out ...')

        while time.time() - start_time < self.time_out_threshold_value:
            if not cart_ref.get():
                print(f"✔️ Congratulations You have successfully checkout the items, Receipt will be mailed to you. Thanks you!")
                return True
            time.sleep(5)

        print(f"Sorry it's ⏰Timeout reached. Session Ended!")
        cart_ref.delete()

    def video_streaming(self):
        if not self.user_id or not self.video_path or not cv2.VideoCapture(self.video_path).isOpened():
            print("❌ Invalid or empty video path or user is not defined. Exiting function.")
            return

        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pred_class = self.predict_frame(frame)
            status = self.custom_push(pred_class)
            if not status:
                cart_ref = db.reference(f'/cart/{self.user_id}')
                if cart_ref.get():
                    cart_ref.delete()
                return

            cv2.putText(frame, f"Prediction: {pred_class}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(f'Vision Transformer Prediction for id {self.user_id}', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cart_ref = db.reference(f'/cart/{self.user_id}')
                if cart_ref.get():
                    cart_ref.delete()
                return

        cap.release()
        cv2.destroyAllWindows()
        self.wait_until_cart_deleted()

