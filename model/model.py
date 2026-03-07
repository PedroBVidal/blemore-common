import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfigurableLinearNN(nn.Module):
    def __init__(self, input_dim, output_dim, model_type, n_layers=1, hidden_dim=256):
        """
        A configurable feedforward neural network with up to 2 hidden layers.

        Args:
            input_dim (int): Size of the input features.
            output_dim (int): Size of the output (number of classes).
            n_layers (int): 0, 1, or 2 hidden layers.
            hidden_dim (int): Number of units in each hidden layer.
        """

        self.model_type = model_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        super(ConfigurableLinearNN, self).__init__()

        if n_layers not in {0, 1}:
            raise ValueError("n_layers must be 0, or 1")

        layers = []
        if n_layers == 0:
            # No hidden layers: input → output
            layers.append(nn.Linear(input_dim, output_dim))

        elif n_layers == 1:
            # One hidden layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.fc = nn.Sequential(*layers)

    def forward(self, x, targets=None):
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)

        loss = None
        if targets is not None:
            log_probs = probs.clamp(min=1e-8).log()
            loss = F.kl_div(log_probs, targets, reduction="batchmean")

        return probs, logits, loss
    



class Backbone_with_ConfigurableLinearNN(nn.Module):
    def __init__(self, backbone, input_dim, output_dim, model_type, n_layers=1, hidden_dim=256):
        """
        A configurable feedforward neural network with up to 2 hidden layers.

        Args:
            input_dim (int): Size of the input features.
            output_dim (int): Size of the output (number of classes).
            n_layers (int): 0, 1, or 2 hidden layers.
            hidden_dim (int): Number of units in each hidden layer.
        """

        self.model_type = model_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        super(Backbone_with_ConfigurableLinearNN, self).__init__()
        self.backbone = backbone

        if n_layers not in {0, 1}:
            raise ValueError("n_layers must be 0, or 1")

        layers = []
        if n_layers == 0:
            # No hidden layers: input → output
            layers.append(nn.Linear(input_dim, output_dim))

        elif n_layers == 1:
            # One hidden layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.fc = nn.Sequential(*layers)

    def forward(self, x, targets=None):
        backbone_embedd = self.backbone(x)
        # Check for 'dead' embeddings
        # We calculate the average magnitude of the vectors in this batch
        avg_magnitude = torch.mean(torch.abs(backbone_embedd)).item()
        if avg_magnitude < 1e-5:
            print(f"⚠️ Warning: Embeddings are nearly zero (Mag: {avg_magnitude:.8f}). Check padding!")
    

        # logits = self.fc(x)
        logits = self.fc(backbone_embedd)
        log_probs= F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        loss = None
        if targets is not None:
            #log_probs = probs.clamp(min=1e-8).log()
            loss = F.kl_div(log_probs, targets, reduction="batchmean")

        return probs, logits, loss