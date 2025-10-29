from PSActivation import PsActivation
from torch.nn import *
from torch.nn.functional import mse_loss
from transformers.activations import (
    GELUActivation,
    NewGELUActivation,
    QuickGELUActivation,
)
from utils import *


import torch.nn as nn
import matplotlib.pyplot as plt

# Mapping of activation names to their corresponding classes
activation_mapping = {
    "ReLU": ReLU,
    "GELU": GELU,
    "SiLU": SiLU,
    "GELUActivation": GELUActivation,
    "NewGELUActivation": NewGELUActivation,
    "QuickGELUActivation": QuickGELUActivation,
    "Tanh": Tanh,
    "Sigmoid": Sigmoid,
    "Softmax": Softmax,
    "Softplus": Softplus,
}

def replace_act(module, act_name, dy, l, r):
    cls = activation_mapping.get(act_name, None)

    # Iterate through child modules and replace activations
    for name, child in module.named_children():
        if isinstance(child, cls):
            setattr(
                module,
                name,
                PsActivation(act_name, dy, l, r)
            )
        else:
            replace_act(child, act_name, dy, l, r)


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

    cls = globals()[activation_name.lower()]

    replace_act(n, activation_name, 0.005, -5, 25)

    # Create a linearly spaced tensor for input values
    x = torch.linspace(-2, 2, 10)

    # Calculate the output using the approximated PsActivation
    yhat = n(x)

    ys = cls(x).clone().detach()

    # Calculate the mean squared error between the real and approximated outputs
    mse = mse_loss(yhat, ys)
    print(f"MSE: {mse}")

    # Plot the real and approximated outputs for comparison
    plt.plot(x, ys, label="real")
    plt.plot(x, yhat, label="approx")
    plt.legend()
    plt.show()
