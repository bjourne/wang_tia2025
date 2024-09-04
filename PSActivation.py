import torch
import torch.nn as nn
from ps_coding import ps  # Import custom ps function for spiking neural network coding
from prepare_hdT import (
    find_hdT,
)  # Import function to find and prepare the high-dimensional T values
from transformers.activations import (
    GELUActivation,
    NewGELUActivation,
    QuickGELUActivation,
)
import json

# Load configuration settings from a JSON file
with open("config.json", "r") as f:
    config = json.load(f)
hdT_path = config["hdT_path"]  # Path for high-dimensional T values

# Mapping of activation names to their corresponding classes
activation_mapping = {
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
    "GELUActivation": GELUActivation,
    "NewGELUActivation": NewGELUActivation,
    "QuickGELUActivation": QuickGELUActivation,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "Softmax": nn.Softmax,
    "Softplus": nn.Softplus,
}


class PsActivation(nn.Module):
    """
    Custom activation module that uses PS coding for spiking neural networks.
    """

    def __init__(self, activation_name, device="cuda", dy=None, l=None, r=None):
        super().__init__()
        activation_name = (
            activation_name.lower()
        )  # Convert activation name to lowercase
        if activation_name == "geluactivation":
            activation_name = "gelu"  # Adjust name for consistency

        # Determine parameter file path based on input configuration
        if dy is not None:
            self.param_path = f"{hdT_path}/{activation_name}_{dy}_{l}_{r}.pt"
        else:
            self.param_path = f"{hdT_path}/{activation_name}.pt"

        # Load high-dimensional T parameters; if unavailable, generate them
        try:
            self.hdT = torch.load(self.param_path)
        except:
            find_hdT(activation_name, dy, l, r)  # Generate and prepare hdT
            self.hdT = torch.load(self.param_path)

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
        print(f"using ps activation: {activation_name}, dy: {dy}, l: {l}, r: {r}")
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


def replace_activation_with_Psactivation(
    module, activation_name, device="cuda", dy=None, l=None, r=None
):
    """
    Replace standard activations with PS activations in a neural network module.
    """
    # Get the activation class based on the activation name
    activation_class = activation_mapping.get(activation_name, None)
    if activation_class is None:
        raise ValueError(f"Activation function {activation_name} is not recognized.")

    # Iterate through child modules and replace activations
    for name, child in module.named_children():
        if isinstance(child, activation_class):
            setattr(
                module,
                name,
                PsActivation(activation_name, device=device, dy=dy, l=l, r=r),
            )
        else:
            # Recursively replace activations in nested modules
            replace_activation_with_Psactivation(
                child, activation_name, device=device, dy=dy, l=l, r=r
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
