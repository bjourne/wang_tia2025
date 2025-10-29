import json
import torch
import torch.nn as nn


from pathlib import Path
from prepare_hdT import find_hdT
from torch.nn import *

from utils import *


def ps(x, params, idx):
    """
    Perform a spiking operation on the input tensor.

    Args:
        x (torch.Tensor): The input tensor.
        h (torch.Tensor): Tensor containing high-dimensional values.
        d (torch.Tensor): Tensor containing step sizes for output.
        T (torch.Tensor): Tensor containing threshold values.
        b (float): Offset to be subtracted from the output.
        idx (torch.Tensor): Indices for selecting elements from h.

    Returns:
        out (torch.Tensor): The output tensor after applying the spiking operation.
        spikes (int): The total number of spikes (activation events) that occurred.
    """
    h, d, T, b = params["h"], params["d"], params["T"], params["b"]

    print("T", T)
    print("d", d)

    print(x.shape)

    v = x.clone()  # Clone input tensor to maintain original values

    # Init to negative bias
    out = torch.zeros_like(x) - b

    K = len(d) - 1  # Determine the number of steps

    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)

    for t in range(1, len(d)):
        print(t, v)
        out += (v - T[t] >= 0).float() * d[t]

        if t != K:
            # Update input for next time step
            v = h[idx, t + 1]

    return out



# Load configuration settings from a JSON file
with open("config.json", "r") as f:
    config = json.load(f)
hdT_path = config["hdT_path"]  # Path for high-dimensional T values



class PsActivation(Module):
    def __init__(self, act_name, dy, l, r):
        super().__init__()
        act_name = act_name.lower()

        # Determine parameter file path based on input configuration
        if dy is not None:
            fname = f"{act_name}_{dy}_{l}_{r}.pt"
        else:
            fname = f"{act_name}.pt"
        self.param_path = Path(hdT_path) / fname

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

    def reset_count(self):
        """
        Reset spike and neuron count to zero.
        """
        self.spike_count = 0
        self.neruon_count = 0

    def forward(self, x, *args, **kwargs):
        """
        Forward pass through the PS activation.
        """
        sp = x.shape  # Store original shape of input
        x_flat = x.view(-1)  # Flatten input for processing
        # Get the first column of h as reference points
        _h = self.hdT["h"][:, 0]

        # Find indices of closest points in h to elements in x
        idx = torch.searchsorted(_h, x_flat)
        idx = torch.clamp(idx, 1, len(_h) - 1)  # Ensure indices are within valid range

        # Determine nearest points
        left = _h[idx - 1]
        right = _h[idx]

        # Calculate differences to nearest points
        left_diff = torch.abs(x_flat - left)
        right_diff = torch.abs(x_flat - right)

        # Select the nearest point based on differences
        nearest = torch.where(left_diff < right_diff, left, right)
        nearest_idx = torch.where(left_diff < right_diff, idx - 1, idx)

        # Reshape to original shape
        x = nearest.view(sp)
        idx = nearest_idx.view(sp)

        # Apply ps function to compute output and spikes
        return ps(
            x,
            self.hdT,
            idx
        )
