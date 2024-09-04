import torch

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
