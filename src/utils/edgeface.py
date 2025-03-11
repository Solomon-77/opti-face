import timm  # Detailed: Import the TIMM library, which provides access to many pre-trained vision models.
           # Simple: Lets us load pre-trained image models.
import torch.nn as nn  # Detailed: Import PyTorch's neural network module (nn) for building layers and models.
                      # Simple: Helps us build neural network layers.

class LoRaLin(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        # Detailed: Initialize a low-rank linear layer that approximates a standard linear layer using two smaller linear layers.
        # Simple: This replaces one big linear layer with two smaller ones.
        super().__init__()  # Detailed: Call the constructor of the parent class (nn.Module) to properly initialize the module.
                           # Simple: Initialize the base module.
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        # Detailed: Define the first linear transformation that reduces the dimensionality from in_features to a lower rank; no bias is used.
        # Simple: First small layer: from input size to rank (no bias).
        self.linear2 = nn.Linear(rank, out_features, bias=bias)
        # Detailed: Define the second linear transformation that maps the lower-dimensional representation to out_features, with optional bias.
        # Simple: Second small layer: from rank to output size, with bias if needed.

    def forward(self, x):
        # Detailed: Define the forward pass where the input is passed sequentially through the two linear layers.
        # Simple: Process input through the two layers.
        return self.linear2(self.linear1(x))
        # Detailed: First apply linear1 to reduce dimensionality, then linear2 to produce the final output.
        # Simple: Output = second_layer(first_layer(input)).

def replace_linear_with_lowrank(model, rank_ratio=0.2):
    # Detailed: Recursively traverse the model's children and replace every nn.Linear layer (except those containing 'head' in their name) with a LoRaLin layer.
    # Simple: Go through the model and swap big linear layers with our two-layer version.
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and 'head' not in name:
            # Detailed: Check if the current module is a linear layer and ensure it's not a classifier head (often named 'head').
            # Simple: If it's a linear layer (not the head), replace it.
            rank = max(2, int(min(module.in_features, module.out_features) * rank_ratio))
            # Detailed: Compute the new rank as a fraction (rank_ratio) of the smaller dimension (in_features or out_features), ensuring a minimum rank of 2.
            # Simple: Determine the new rank (at least 2) based on rank_ratio.
            bias = module.bias is not None
            # Detailed: Determine whether the original linear layer uses a bias.
            # Simple: Check if bias exists.
            setattr(model, name, LoRaLin(module.in_features, module.out_features, rank, bias))
            # Detailed: Replace the original linear layer with a new LoRaLin layer that has the same input and output dimensions, using the computed rank and bias setting.
            # Simple: Swap the big layer with our low-rank version.
        else:
            replace_linear_with_lowrank(module, rank_ratio)
            # Detailed: Recursively apply the replacement to child modules if the current module is not a linear layer.
            # Simple: Look inside this module for more linear layers.
    return model
    # Detailed: Return the modified model with all applicable linear layers replaced.
    # Simple: Give back the updated model.

class TimmFRWrapperV2(nn.Module):
    def __init__(self, model_name='edgenext_x_small', featdim=512):
        # Detailed: Initialize a wrapper for a TIMM face recognition model. The model is created and its classifier is reset to output a feature vector of size featdim.
        # Simple: Set up a face recognition model that outputs a feature vector of the desired size.
        super().__init__()  # Detailed: Initialize the base nn.Module.
                           # Simple: Initialize the module.
        self.model = timm.create_model(model_name, pretrained=True)
        # Detailed: Create the specified model using timm.create_model with pretrained weights.
        # Simple: Load the pre-trained model by its name.
        self.model.reset_classifier(featdim)
        # Detailed: Replace the default classifier head of the model with one that outputs featdim features.
        # Simple: Change the final layer so it outputs the desired feature size.

    def forward(self, x):
        # Detailed: Forward the input x through the wrapped TIMM model.
        # Simple: Pass the input through the model.
        return self.model(x)
        # Detailed: Return the model's output.
        # Simple: Output the result.

def get_model(name):
    # Detailed: Factory function that returns a face recognition model based on the provided model name.
    # Simple: Choose and build a model based on the given name.
    if name == 'edgeface_xs_gamma_06':
        # Detailed: For the model 'edgeface_xs_gamma_06', create a TimmFRWrapperV2 with the 'edgenext_x_small' architecture, then replace its linear layers with low-rank approximations using a rank ratio of 0.6.
        # Simple: For edgeface_xs_gamma_06, use a small model with 60% low-rank reduction.
        return replace_linear_with_lowrank(TimmFRWrapperV2('edgenext_x_small'), rank_ratio=0.6)
    elif name == 'edgeface_s_gamma_05':
        # Detailed: For the model 'edgeface_s_gamma_05', create a TimmFRWrapperV2 with the 'edgenext_small' architecture, then replace its linear layers with low-rank approximations using a rank ratio of 0.5.
        # Simple: For edgeface_s_gamma_05, use a small model with 50% low-rank reduction.
        return replace_linear_with_lowrank(TimmFRWrapperV2('edgenext_small'), rank_ratio=0.5)
    else:
        # Detailed: If the provided model name is not recognized, raise a ValueError.
        # Simple: If the model name isn’t known, show an error.
        raise ValueError(f"Unknown model name: {name}")
