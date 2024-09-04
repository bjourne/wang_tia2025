from PSActivation import (
    replace_activation_with_Psactivation,
)  # Import the function to replace activation with PsActivation
from utils import *
import torch.nn as nn
import matplotlib.pyplot as plt


# Define a neural network model with a single GELU activation
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.act = nn.GELU()  # Initialize with GELU activation

    def forward(self, x):
        x = self.act(x)  # Apply GELU activation
        return x  # Return the activated input


if __name__ == "__main__":
    n = net()  # Instantiate the neural network model
    activation_name = "GELU"

    # Replace GELU activation with PsActivation in the model
    replace_activation_with_Psactivation(
        n, activation_name, "cpu", dy=0.001, l=-5, r=25
    )

    # Create a linearly spaced tensor for input values
    x = torch.linspace(-25, 25, 100000)

    # Calculate the output using the approximated PsActivation
    y_gelu_approx = n(x)

    # Calculate the output using the real GELU function
    y_gelu_real = globals()[activation_name.lower()](x).clone().detach()

    # Calculate the mean squared error between the real and approximated outputs
    mse = torch.mean((y_gelu_real - y_gelu_approx) ** 2)
    print(f"MSE: {mse}")  # Print the mean squared error

    # Plot the real and approximated outputs for comparison
    plt.plot(x, y_gelu_real, label="real")
    plt.plot(x, y_gelu_approx, label="approx")
    plt.legend()
    plt.show()
