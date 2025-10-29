#from PSActivation import PsActivation

from prepare_hdT import find_hdT
from torch.nn import *
from torch.nn.functional import mse_loss
from transformers.activations import (
    GELUActivation,
    NewGELUActivation,
    QuickGELUActivation,
)
from utils import *


import json
import torch.nn as nn
import matplotlib.pyplot as plt

# Load configuration settings from a JSON file
with open("config.json", "r") as f:
    config = json.load(f)

class PsActivation(Module):
    def __init__(self, act_name, dy, l, r):
        super().__init__()
        act_name = act_name.lower()

        # Determine parameter file path based on input configuration
        if dy is not None:
            fname = f"{act_name}_{dy}_{l}_{r}.pt"
        else:
            fname = f"{act_name}.pt"
        self.param_path = fname

        # Load high-dimensional T parameters; if unavailable, generate them
        try:
            self.hdT = torch.load(self.param_path, weights_only = False)
        except Exception as err:
            print("error", err)
            # Generate and prepare hdT
            find_hdT(act_name, dy, l, r)
            self.hdT = torch.load(self.param_path, weights_only = False)

        # Extract dy, l, r from hdT configuration
        dy = self.hdT["dy"]
        l = self.hdT["l"]
        r = self.hdT["r"]

        # Display activation settings and parameters
        print(f"using ps activation: {act_name}, dy: {dy}, l: {l}, r: {r}")

    def forward(self, x, *args, **kwargs):
        """
        Forward pass through the PS activation.
        """
        sp = x.shape  # Store original shape of input

        # Get the first column of h as reference points
        _h = self.hdT["h"][:, 0]

        # Find indices of closest points in h to elements in x
        idx = torch.searchsorted(_h, x)
        idx = torch.clamp(idx, 1, len(_h) - 1)  # Ensure indices are within valid range

        # Determine nearest points
        left = _h[idx - 1]
        right = _h[idx]

        # Calculate differences to nearest points
        go_left = torch.abs(x - left) < torch.abs(x - right)

        # Select the nearest point based on differences
        nearest = torch.where(go_left, left, right)
        nearest_idx = torch.where(go_left, idx - 1, idx)

        # Reshape to original shape
        idx = nearest_idx.view(sp)

        quant_x = self.hdT["h"][idx].T
        quant_x[1] = x

        print("quantized", quant_x)

        d = self.hdT["d"]
        T = self.hdT["T"]

        spikes = (quant_x >= T.unsqueeze(1)) * d.unsqueeze(1)
        return torch.sum(spikes, dim = 0) - self.hdT["b"]


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

    act = PsActivation("gelu", 0.005, -5, 25)

    # Create a linearly spaced tensor for input values
    xs = torch.linspace(-2, 2, 10)

    #xs = torch.Tensor([-0.2])

    # Calculate the output using the approximated PsActivation
    yhat = act(xs)

    ys = cls(xs).clone().detach()

    # Calculate the mean squared error between the real and approximated outputs
    mse = mse_loss(yhat, ys)
    print(f"MSE: {mse}", yhat, ys)

    # Plot the real and approximated outputs for comparison
    plt.plot(xs, ys, label="real")
    plt.plot(xs, yhat, label="approx")
    plt.legend()
    plt.show()
