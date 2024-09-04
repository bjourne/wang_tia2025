# PSNeurons

Official PSNeurons github repository. PSNeurons are compatible with Pytorch 2.3.1.

You can use demo.py to get a quick overview of how to use PSNeuron.

# Overview of Key Python Scripts

-   **demo.py**

    In `demo.py`, the simplest usage of a PS neuron is demonstrated. You can easily replace the activation function of an artificial neural network with a PS neuron using the `replace_activation_with_Psactivation` function, effectively converting it into a spiking neural network. The fitting performance of the activation function is visually demonstrated through an image.

-   **config.json**

    The `config.json` file contains some default parameters, where you can configure the default storage path, fitting interval, fitting accuracy, and more.

-   **prepare_hdT.py**

    This script implements the parameter iteration process based on the algorithm described in the paper. The results of the parameter iterations are saved in the default storage path. You can specify any activation function to fit using the `target_func_name` parameter. The `dy`, `l`, and `r` parameters can also be specified simultaneously, though they are optional.

-   **PSActivation.py** 

    This file contains the main code for the `PSNeuron` class, including how to perform the ANN-to-SNN conversion using the iterated parameters.The count of PS neurons and the number of spikes generated are also tracked here.

-   **utils.py** 

    This file includes various tools and commonly used activation functions.

# Steps to Run the Code

As shown in `demo.py`, you can directly use the `replace_activation_with_Psactivation` function from `PSActivation.py` to replace the activation functions in an ANN, enabling the ANN-to-SNN conversion. If the corresponding pre-iterated parameters are not found in the default storage location, the `find_hdT` function from `prepare_hdT.py` will be automatically invoked to generate a set of parameters based on the algorithm and save them. During this process, you can manually control the activation function to be generated, as well as its precision and range, by adjusting `activation_name`, `dy`, `l`, and `r`.

Alternatively, you can directly use `prepare_hdT.py` to generate a different set of parameters and observe the parameter iteration process.

# Requirements

The code runs under Python 3.12.4, and the required dependencies are as follows:

>   matplotlib\==3.9.2
>   	numpy\==2.1.1
>   	scipy\==1.14.1
>   	torch\==2.3.1+cu121
>   	transformers\==4.42.0

# Environment Details

The experiments were conducted on a Windows 11 system using the PyTorch framework with a publicly available pre-trained model. The setup included CUDA Version 12.3 and an NVIDIA GeForce RTX 3050 GPU.