import torch
import torch.nn as nn
import torchvision.transforms as transforms
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Color
from PIL import Image as PILImage
import io
import os

# Define your model class
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 6)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

class FruitClassifierApp(App):
    class_names = {
        0: "Fresh Apple",
        1: "Fresh Banana",
        2: "Fresh Orange",
        3: "Rotten Apple",
        4: "Rotten Banana",
        5: "Rotten Orange",
    }

    def build(self):
        self.load_model()
        self.layout = BoxLayout(orientation='vertical')
        
        # Set the background color of the main layout to black
        with self.layout.canvas.before:
            Color(0, 0, 0, 1)  # Black color
            self.rect = Rectangle(size=self.layout.size, pos=self.layout.pos)
        
        self.image = KivyImage()
        
        # Add the KivyImage widget directly to the main layout
        self.layout.add_widget(self.image)
        
        self.label = Label(text='', size_hint_y=0.1)
        self.button = Button(text='Load Image', on_press=self.load_image)
        
        self.layout.add_widget(self.label)
        self.layout.add_widget(self.button)
        
        return self.layout

    def load_model(self):
        try:
            model_path = 'fruit_classifier_scripted.pt'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file '{model_path}' not found")
            
            self.model = torch.jit.load(model_path, map_location=torch.device('cpu'))
            self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error in load_model: {e}")

    def load_image(self, instance):
        try:
            image_path = self.open_file_dialog()
            if image_path:
                print(f"Image path: {image_path}")
                prediction = self.predict_image(image_path)
                self.display_prediction(image_path, prediction)
        except Exception as e:
            print(f"Error in load_image: {e}")

    def open_file_dialog(self):
        try:
            from tkinter import Tk
            from tkinter.filedialog import askopenfilename
            root = Tk()
            root.withdraw()
            file_path = askopenfilename()
            print(f"Selected file path: {file_path}")
            return file_path
        except Exception as e:
            print(f"Error in open_file_dialog: {e}")

    def predict_image(self, image_path):
        try:
            print(f"Loading image from path: {image_path}")
            image = PILImage.open(image_path).resize((224, 224))
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image_tensor = preprocess(image).unsqueeze(0)
            print(f"Image tensor shape: {image_tensor.shape}")

            with torch.no_grad():
                output = self.model(image_tensor)
                print(f"Model output: {output}")

            predicted_class = torch.argmax(output).item()
            print(f"Predicted class index: {predicted_class}")
            return predicted_class
        except Exception as e:
            print(f"Error in predict_image: {e}")

    def display_prediction(self, image_path, prediction):
        try:
            texture = self.load_image_to_texture(image_path)
            self.image.texture = texture
            self.image.source = image_path
            self.image.reload()

            predicted_class_name = self.get_class_name(prediction)
            self.label.text = f'Predicted class: {predicted_class_name}'
        except Exception as e:
            print(f"Error in display_prediction: {e}")

    def get_class_name(self, class_index):
        try:
            class_name = self.class_names.get(class_index, "Unknown")
            print(f"Class name for index {class_index}: {class_name}")
            return class_name
        except Exception as e:
            print(f"Error in get_class_name: {e}")

    def load_image_to_texture(self, image_path):
        try:
            image = PILImage.open(image_path)
            buf = io.BytesIO()
            image.save(buf, format='png')
            buf.seek(0)
            texture = texture.create(size=(image.width, image.height), colorfmt='rgb')
            texture.blit_buffer(buf.read(), colorfmt='rgb', bufferfmt='ubyte')
            return texture
        except Exception as e:
            print("")

if __name__ == '__main__':
    FruitClassifierApp().run()
