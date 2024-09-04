import torch
import numpy as np
from scipy.optimize import fsolve


def tanh(x):
    return np.tanh(x)


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


def sin(x):
    return np.sin(x)


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def swish(x, beta=1):
    return x * sigmoid(beta * x)


def silu(x):
    return x * sigmoid(x)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def inverse_f(f, y, b):
    return torch.tensor(fsolve(lambda x: f(x) + b - y, 0)[0])


def select_elements_by_step(y, dy):
    indices = []
    values = []
    last_value = y[0]
    values.append(last_value)
    indices.append(0)
    for i in range(1, len(y)):
        value = y[i]
        distance = abs(value - last_value)
        if distance >= dy:
            values.append(value)
            indices.append(i)
            last_value = value
    indices = np.array(indices)
    values = np.array(values)
    return indices, values
