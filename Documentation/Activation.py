import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def gaussian(x):
    return np.exp(-x**2)

# Generate x values
x = np.linspace(-10, 10, 400)

# Compute y values for each activation function
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_gaussian = gaussian(x)

# Plot the activation functions
plt.figure(figsize=(10, 6))
plt.plot(x, y_sigmoid, label='Sigmoid', color='blue')
plt.plot(x, y_tanh, label='Tanh', color='green')
plt.plot(x, y_relu, label='ReLU', color='red', linewidth=3, linestyle='--')  # Thicker line and dashed style
plt.plot(x, y_leaky_relu, label='Leaky ReLU', color='purple')
plt.plot(x, y_gaussian, label='Gaussian', color='brown')

# Customize the plot
plt.title('Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.show()
