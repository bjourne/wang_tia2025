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