from PSActivation import replace_activation_with_Psactivation
from torch.nn import *
from torch.nn.functional import mse_loss
from utils import *


import torch.nn as nn
import matplotlib.pyplot as plt


# Define a neural network model with a single GELU activation
class net(Module):
    def __init__(self):
        super().__init__()
        self.act = GELU()

    def forward(self, x):
        return self.act(x)


if __name__ == "__main__":
    n = net()  # Instantiate the neural network model
    activation_name = "GELU"

    # Replace GELU activation with PsActivation in the model
    replace_activation_with_Psactivation(
        n, activation_name, "cpu", 0.005, -5, 25
    )

    # Create a linearly spaced tensor for input values
    x = torch.linspace(-2, 2, 100000)

    # Calculate the output using the approximated PsActivation
    yhat = n(x)

    # Calculate the output using the real GELU function
    #ys = GELU()(x)
    ys = globals()[activation_name.lower()](x).clone().detach()

    # Calculate the mean squared error between the real and approximated outputs
    mse = mse_loss(yhat, ys)
    print(f"MSE: {mse}")

    # Plot the real and approximated outputs for comparison
    plt.plot(x, ys, label="real")
    plt.plot(x, yhat, label="approx")
    plt.legend()
    plt.show()
