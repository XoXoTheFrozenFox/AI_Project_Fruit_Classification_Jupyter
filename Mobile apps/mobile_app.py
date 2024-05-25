import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.graphics import Rectangle, Color
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
import io
import os

# Define your model class
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 6)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(F.max_pool2d(self.relu(self.bn1(self.conv1(x))), 2))
        x = self.dropout(F.max_pool2d(self.relu(self.bn2(self.conv2(x))), 2))
        x = self.dropout(F.max_pool2d(self.relu(self.bn3(self.conv3(x))), 2))
        x = self.dropout(F.max_pool2d(self.relu(self.bn4(self.conv4(x))), 2))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
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
            model_path_scripted = 'fruit_classifier_scripted.pt'
            model_path_pth = 'fruit_classifier.pth'
            
            if not os.path.exists(model_path_scripted):
                if not os.path.exists(model_path_pth):
                    raise FileNotFoundError(f"Model file '{model_path_pth}' not found")
                
                # Load the .pth model
                model = Net()
                model.load_state_dict(torch.load(model_path_pth, map_location=torch.device('cpu')))
                
                # Script the model
                scripted_model = torch.jit.script(model)
                
                # Save the scripted model
                torch.jit.save(scripted_model, model_path_scripted)
                print(f"Scripted model saved to '{model_path_scripted}'")

            # Load the scripted model
            self.model = torch.jit.load(model_path_scripted, map_location=torch.device('cpu'))
            self.model.eval()
            print("Model loaded successfully")
        
        except FileNotFoundError as fnf_error:
            print(f"File not found error: {fnf_error}")
        except Exception as e:
            print(f"Error in load_model: {e}")

    def load_image(self, instance):
        try:
            image_path = self.open_file_dialog()
            if image_path:
                print(f"Image path: {image_path}")
                prediction, preprocessed_image = self.predict_image(image_path)
                self.display_prediction(image_path, prediction)
                self.show_preprocessed_image(preprocessed_image)
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
            return predicted_class, image_tensor
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
            texture = CoreImage(buf, ext='png').texture
            return texture
        except Exception as e:
            print(f"Error in load_image_to_texture: {e}")

    def show_preprocessed_image(self, preprocessed_image):
        try:
            # Convert the tensor to a PIL image
            unloader = transforms.ToPILImage()
            image = preprocessed_image.squeeze(0)  # Remove the batch dimension
            image = unloader(image)

            # Save the preprocessed image to a buffer
            buf = io.BytesIO()
            image.save(buf, format='png')
            buf.seek(0)
            texture = CoreImage(buf, ext='png').texture

            # Create a new Popup window to show the preprocessed image
            popup_layout = BoxLayout(orientation='vertical')
            popup_image = KivyImage(texture=texture)
            close_button = Button(text="Close", size_hint_y=0.1, on_press=self.close_popup)

            popup_layout.add_widget(popup_image)
            popup_layout.add_widget(close_button)

            #self.popup = Popup(title='Preprocessed Image', content=popup_layout, size_hint=(0.8, 0.8))
            #self.popup.open()
        except Exception as e:
            print(f"Error in show_preprocessed_image: {e}")

    def close_popup(self, instance):
        if hasattr(self, 'popup'):
            self.popup.dismiss()

if __name__ == '__main__':
    FruitClassifierApp().run()
