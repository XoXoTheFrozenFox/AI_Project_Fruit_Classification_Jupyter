import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the image using PIL and resize it
image_path = 'C:/Users/S_CSIS-PostGrad/Downloads/ergwerf.jpg'  # Specify your image path
img = Image.open(image_path)
img_resized = img.resize((224, 224), Image.LANCZOS)  # Use LANCZOS for high-quality downsampling

# Convert the resized image to a format that matplotlib can use
img_resized = np.array(img_resized)

# Create a figure and axis
fig, ax = plt.subplots()

# Display the resized image
ax.imshow(img_resized)

# Set the limits of the x and y axis to 224
ax.set_xlim(0, 224)
ax.set_ylim(224, 0)  # Invert the y-axis

# Add labels to the axes
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')

# Add a title at the top of the image
ax.set_title('label = Rotten Orange', fontsize=16, pad=20)

# Show the plot
plt.show()
