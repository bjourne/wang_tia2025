import json
import torch
import torch.nn as nn


from pathlib import Path
#from ps_coding import ps  # Import custom ps function for spiking neural network coding
from prepare_hdT import find_hdT
from torch.nn import *
from transformers.activations import (
    GELUActivation,
    NewGELUActivation,
    QuickGELUActivation,
)

from utils import *  # Import utility functions from utils module


def ps(x, h, d, T, b, idx):
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
    v = x.clone()  # Clone input tensor to maintain original values
    z = torch.zeros_like(x)  # Initialize tensor to track spikes
    out = torch.zeros_like(x)  # Initialize output tensor
    t = 1  # Initialize time step counter
    K = len(d) - 1  # Determine the number of steps
    spikes = 0  # Initialize spike counter

    while t <= K:
        # Determine where input exceeds threshold and create spike indicators
        z = torch.where(
            v - T[t] >= 0,  # Compare input to threshold
            torch.ones_like(v),  # Set 1 where condition is met
            torch.zeros_like(v),  # Set 0 where condition is not met
        )
        out += z * d[t]  # Add step size to output where spikes occur
        spikes += z.sum()  # Count total spikes

        if t != K:
            # Update input for next time step
            v = h[idx, t + 1]

        t += 1  # Move to next time step

    out -= b  # Subtract the offset from the final output
    return out, spikes  # Return the output and the spike count



# Load configuration settings from a JSON file
with open("config.json", "r") as f:
    config = json.load(f)
hdT_path = config["hdT_path"]  # Path for high-dimensional T values

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


class PsActivation(Module):
    def __init__(self, act_name, device, dy, l, r):
        super().__init__()
        act_name = act_name.lower()
        if act_name == "geluactivation":
            act_name = "gelu"  # Adjust name for consistency

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

        # Load model parameters
        self.h = self.hdT["h"].to(device)
        self.d = self.hdT["d"].to(device)
        self.T = self.hdT["T"].to(device)
        self.b = self.hdT["b"]

        # Initialize spike and neuron count
        self.spike_count = 0
        self.neruon_count = 0

        # Extract dy, l, r from hdT configuration
        dy = self.hdT["dy"]
        l = self.hdT["l"]
        r = self.hdT["r"]

        # Display activation settings and parameters
        print(f"using ps activation: {act_name}, dy: {dy}, l: {l}, r: {r}")
        print(f"numh: {self.h.shape[0]}, K: {self.h.shape[1]-1}")

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
        _h = self.h[:, 0]  # Get the first column of h as reference points

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
        out, spikes = ps(x, self.h, self.d, self.T, self.b, idx)

        # Update spike and neuron count
        self.spike_count += spikes
        self.neruon_count += x.numel()

        return out


def replace_activation_with_Psactivation(module, act_name, device, dy, l, r):
    print("Working...")
    # Get the activation class based on the activation name
    activation_class = activation_mapping.get(act_name, None)
    if activation_class is None:
        raise ValueError(f"Activation function {act_name} is not recognized.")

    # Iterate through child modules and replace activations
    for name, child in module.named_children():
        if isinstance(child, activation_class):
            setattr(
                module,
                name,
                PsActivation(act_name, device, dy, l, r)
            )
        else:
            # Recursively replace activations in nested modules
            replace_activation_with_Psactivation(
                child, act_name, device, dy=dy, l=l, r=r
            )


def count_neurons_and_spikes(model):
    """
    Count the total number of neurons and spikes in a model with PS activations.
    """
    spike_neuron_count = 0
    total_spike_count = 0

    # Iterate through model layers and accumulate counts from PS activations
    for layer in model.modules():
        if isinstance(layer, PsActivation):
            spike_neuron_count += layer.neruon_count
            total_spike_count += layer.spike_count
            layer.reset_count()  # Reset count after reading

    return spike_neuron_count, total_spike_count
