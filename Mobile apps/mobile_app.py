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
from kivy.metrics import dp
from kivy.uix.popup import Popup
from PIL import Image as PILImage
from bs4 import BeautifulSoup
import requests
import io
import os
import re

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

# Define the welcome panel
class WelcomePanel(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint_y = 0.2
        welcome_label = Label(
            text="Welcome to the Fruit Classifier App!",
            font_size=24,
            color=(1, 1, 1, 1)
        )
        self.add_widget(welcome_label)

# Define the main app class
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
        root_layout = BoxLayout(orientation='vertical')

        # Add welcome panel
        welcome_panel = WelcomePanel()
        root_layout.add_widget(welcome_panel)

        # Add image display and buttons
        self.image = KivyImage(source='C:/Users/Nico/Downloads/background.png')
        root_layout.add_widget(self.image)
        self.label = Label(text='', size_hint_y=0.1)

        # Load Image Button
        self.button = Button(
            text='Load Image',
            size_hint=(None, None),
            size=(dp(120), dp(40)),
            on_press=self.load_image
        )
        self.button.border_radius = [15]
        self.button.background_color = (0.2, 0.7, 0.3, 1)
        button_layout = BoxLayout(size_hint_y=0.1, padding=(dp(20), 0))
        button_layout.add_widget(self.button)

        # More Information Button
        self.btn1 = Button(
            text='More information',
            size_hint=(None, None),
            size=(dp(120), dp(40)),
            on_press=self.on_more_info_click
        )
        self.btn1.border_radius = [15]
        self.btn1.background_color = (0.2, 0.7, 0.3, 1)
        moreinfo_layout = BoxLayout(size_hint_y=0.1, padding=(dp(650), 40))
        moreinfo_layout.add_widget(self.btn1)

        # Add widgets to root layout
        root_layout.add_widget(self.label)
        root_layout.add_widget(button_layout)
        root_layout.add_widget(moreinfo_layout)

        return root_layout

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

            self.predicted_class_name = self.get_class_name(prediction)
            self.label.text = f'Predicted class: {self.predicted_class_name}'
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
            image = preprocessed_image.squeeze(0) # Remove the batch dimension
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

            self.popup = Popup(title='Preprocessed Image', content=popup_layout, size_hint=(0.8, 0.8))
            self.popup.open()
        except Exception as e:
            print(f"Error in show_preprocessed_image: {e}")


    def clean_prediction(self, prediction):
        """
        Remove the words 'fresh' or 'rotten' from the prediction.
        """
        cleaned_prediction = re.sub(r'\b(?:fresh|rotten)\b', '', prediction, flags=re.IGNORECASE).strip()
        return cleaned_prediction
    
    def on_more_info_click(self, instance):
        try:
            if hasattr(self, 'predicted_class_name'):
                cleaned_prediction = self.clean_prediction(self.predicted_class_name)
                if 'Rotten' in self.predicted_class_name:
                    self.show_rotten_popup()
                else:
                    calories = self.fetch_calories(cleaned_prediction)
                    self.show_calories_popup(calories)
            else:
                self.show_no_fruit_detected_popup()
        except Exception as e:
            print(f"Error in on_more_info_click: {e}")

    def fetch_calories(self, prediction):
        try:
            url = f'https://www.google.com/search?&q=calories in {prediction}'
            req = requests.get(url).text
            scrap = BeautifulSoup(req, 'html.parser')
            calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
            return calories
        except Exception as e:
            print(f"Can't fetch the Calories: {e}")
            return "Unable to fetch calories information"
        
    def show_rotten_popup(self):
        try:
            popup_layout = BoxLayout(orientation='vertical', padding=10)
            message_label = Label(text="Sorry, no information available on rotten fruit.")
            close_button = Button(text="Close", size_hint_y=0.2, on_press=self.close_popup)

            popup_layout.add_widget(message_label)
            popup_layout.add_widget(close_button)

            self.popup = Popup(title='Information', content=popup_layout, size_hint=(0.6, 0.4))
            self.popup.open()
        except Exception as e:
            print(f"Error in show_rotten_popup: {e}")

    def show_calories_popup(self, calories):
        try:
            popup_layout = BoxLayout(orientation='vertical', padding=10)
            calories_label = Label(text=f'Calories: {calories}', font_size=24)
            close_button = Button(text="Close", size_hint_y=0.2, on_press=self.close_popup)

            popup_layout.add_widget(calories_label)
            popup_layout.add_widget(close_button)

            self.popup = Popup(title='Calories Information', content=popup_layout, size_hint=(0.6, 0.4))
            self.popup.open()
        except Exception as e:
            print(f"Error in show_calories_popup: {e}")

    def show_no_fruit_detected_popup(self):
        try:
            popup_layout = BoxLayout(orientation='vertical', padding=10)
            message_label = Label(text="No fruit detected.")
            close_button = Button(text="Close", size_hint_y=0.2, on_press=self.close_popup)

            popup_layout.add_widget(message_label)
            popup_layout.add_widget(close_button)

            self.popup = Popup(title='Information', content=popup_layout, size_hint=(0.6, 0.4))
            self.popup.open()
        except Exception as e:
            print(f"Error in show_no_fruit_detected_popup: {e}")


    def close_popup(self, instance):
        self.popup.dismiss()

# Run the app
if __name__ == '__main__':
    FruitClassifierApp().run()
