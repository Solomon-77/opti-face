import timm  # Detailed: Import the TIMM library, which provides access to a large collection of pre-trained vision models.
           # Simple: Use TIMM to load pre-trained vision models.
import torch.nn as nn  # Detailed: Import PyTorch's neural network module as nn for building layers and models.
                      # Simple: Lets us build neural networks.

class LoRaLin(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        # Detailed: Initialize a low-rank linear layer module that factorizes a standard linear layer into two smaller linear layers.
        # Simple: This replaces one big linear layer with two smaller ones.
        super().__init__()  # Detailed: Call the constructor of the parent class (nn.Module).
                           # Simple: Initialize the base class.
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        # Detailed: Define the first linear transformation, reducing the input from in_features to a lower-dimensional space (rank) without using a bias.
        # Simple: First small layer: from input size to rank (no bias).
        self.linear2 = nn.Linear(rank, out_features, bias=bias)
        # Detailed: Define the second linear transformation that maps the low-dimensional representation (rank) to the desired out_features, optionally using a bias.
        # Simple: Second small layer: from rank to output size, with bias if needed.

    def forward(self, x):
        # Detailed: Compute the forward pass by first applying linear1 to the input and then linear2 to the result.
        # Simple: Pass input through both small layers in sequence.
        return self.linear2(self.linear1(x))
        # Detailed: Return the output after both linear transformations.
        # Simple: Output = second layer(first layer(input)).

def replace_linear_with_lowrank(model, rank_ratio=0.2):
    # Detailed: Recursively traverse the model's submodules and replace each nn.Linear (excluding those with 'head' in their name) with a LoRaLin module to approximate a low-rank version.
    # Simple: Go through the model and swap big linear layers with our two-layer version.
    for name, module in model.named_children():
        # Detailed: Iterate over each child module of the current model, obtaining both its name and the module itself.
        # Simple: Look at each part of the model.
        if isinstance(module, nn.Linear) and 'head' not in name:
            # Detailed: Check if the current module is a linear layer and ensure its name does not include 'head' (to avoid replacing classifier heads).
            # Simple: If it's a linear layer and not the final head, then replace it.
            rank = max(2, int(min(module.in_features, module.out_features) * rank_ratio))
            # Detailed: Calculate the new rank as a fraction (rank_ratio) of the smaller dimension between in_features and out_features, ensuring a minimum rank of 2.
            # Simple: Determine the new size (rank), but at least 2.
            bias = module.bias is not None
            # Detailed: Check whether the original linear layer has a bias term.
            # Simple: See if bias is used.
            setattr(model, name, LoRaLin(module.in_features, module.out_features, rank, bias))
            # Detailed: Replace the current linear layer in the model with a new LoRaLin layer that has the same input and output dimensions, and the computed rank and bias settings.
            # Simple: Swap out the big layer with our two-layer version.
        else:
            replace_linear_with_lowrank(module, rank_ratio)
            # Detailed: If the module is not a linear layer or is a container module, recursively apply the replacement function to its submodules.
            # Simple: Check inside this module for more layers to replace.
    return model  # Detailed: Return the modified model with all applicable linear layers replaced by low-rank approximations.
                  # Simple: Give back the updated model.

class TimmFRWrapperV2(nn.Module):
    def __init__(self, model_name='edgenext_x_small', featdim=512):
        # Detailed: Initialize a wrapper module for a TIMM face recognition model. This wrapper loads a pre-trained model and resets its classifier head to output features of dimension featdim.
        # Simple: Set up a face recognition model with a custom feature output size.
        super().__init__()  # Detailed: Call the parent class constructor.
                           # Simple: Initialize the base class.
        self.model = timm.create_model(model_name, pretrained=True)
        # Detailed: Create a model using timm.create_model with the specified model_name and load its pre-trained weights.
        # Simple: Build the model and load pre-trained weights.
        self.model.reset_classifier(featdim)
        # Detailed: Replace the model's default classifier head with a new one that outputs a feature vector of size featdim.
        # Simple: Change the last layer so it outputs our desired feature size.

    def forward(self, x):
        # Detailed: Forward the input through the underlying TIMM model.
        # Simple: Pass the input through the model.
        return self.model(x)
        # Detailed: Return the output generated by the TIMM model.
        # Simple: Output the model's result.

def get_model(name):
    # Detailed: Factory function that selects and returns a face recognition model based on the given name.
    #          It creates a TimmFRWrapperV2 model and replaces its linear layers with low-rank approximations using a specified rank_ratio.
    # Simple: Choose and build a model based on the given name.
    if name == 'edgeface_xs_gamma_06':
        # Detailed: For the model name 'edgeface_xs_gamma_06', create a TimmFRWrapperV2 model using the 'edgenext_x_small' architecture and apply a low-rank replacement with a rank_ratio of 0.6.
        # Simple: For edgeface_xs_gamma_06, use the small model with 60% low-rank reduction.
        return replace_linear_with_lowrank(TimmFRWrapperV2('edgenext_x_small'), rank_ratio=0.6)
    elif name == 'edgeface_s_gamma_05':
        # Detailed: For the model name 'edgeface_s_gamma_05', create a TimmFRWrapperV2 model using the 'edgenext_small' architecture and apply a low-rank replacement with a rank_ratio of 0.5.
        # Simple: For edgeface_s_gamma_05, use the small model with 50% low-rank reduction.
        return replace_linear_with_lowrank(TimmFRWrapperV2('edgenext_small'), rank_ratio=0.5)
    else:
        # Detailed: If the provided model name is not recognized, raise a ValueError indicating an unknown model name.
        # Simple: If the name isn't recognized, show an error.
        raise ValueError(f"Unknown model name: {name}")
