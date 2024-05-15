import torch
import torch.nn as nn
import torch.nn.functional as F  # Import torch.nn.functional as F instead of torchvision.models as F
import tkinter as tk
from tkinter import filedialog
import torchvision.transforms as transforms
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from PIL import Image as PILImage
import io

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

# Instantiate your model
model = Net()
def load_model(self):
    # Load the model onto the CPU
    self.model = torch.jit.load('fruit_classifier_scripted.pt', map_location=torch.device('cpu'))
model.eval()

# Example input
example_input = torch.rand(1, 3, 224, 224)  # Adjust input size as per your model

# Convert model to TorchScript
traced_model = torch.jit.trace(model, example_input)
traced_model.save('C:/CODE/Code/CODE ON GITHUB/AI_Project_Fruit_Classification_Jupyter//fruit_classifier_scripted.pt')

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
        self.image = Image()
        self.button = Button(text='Load Image', on_press=self.load_image)
        self.layout.add_widget(self.image)
        self.layout.add_widget(self.button)
        return self.layout

    def load_model(self):
        self.model = torch.jit.load('fruit_classifier_scripted.pt')

    def load_image(self, instance):
        # Open file dialog to select an image
        image_path = self.open_file_dialog()
        if image_path:
            # Perform inference on the selected image
            prediction = self.predict_image(image_path)
            # Display prediction result
            self.display_prediction(image_path, prediction)

    def open_file_dialog(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        file_path = filedialog.askopenfilename()
        return file_path

    def predict_image(self, image_path):
        # Open and preprocess the image
        image = PILImage.open(image_path).resize((224, 224))
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(image_tensor)

        # Get the predicted class
        predicted_class = torch.argmax(output).item()
        return predicted_class

    def display_prediction(self, image_path, prediction):
        # Display the image along with the prediction
        texture = self.load_image_to_texture(image_path)
        self.image.texture = texture
        self.image.source = image_path  # Set the image source
        self.image.reload()

        # Get the class name corresponding to the predicted class index
        predicted_class_name = self.get_class_name(prediction)

        # Print or display the prediction result
        print(f'Predicted class: {predicted_class_name}')

    def get_class_name(self, class_index):
        return self.class_names.get(class_index, "Unknown")

    def load_image_to_texture(self, image_path):
        # Load image and convert to texture for display in Kivy
        image = PILImage.open(image_path)
        buf = io.BytesIO()
        image.save(buf, format='png')
        buf.seek(0)
        return Texture.create(size=(image.width, image.height), colorfmt='rgba')

if __name__ == '__main__':
    FruitClassifierApp().run()