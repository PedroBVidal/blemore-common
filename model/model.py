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

        # DEFAULT LOSS
        # loss = None
        # if targets is not None:
        #     log_probs = probs.clamp(min=1e-8).log()
        #     loss = F.kl_div(log_probs, targets, reduction="batchmean")

        # LABEL_TO_INDEX = {
        #     "ang": 0,
        #     "disg": 1,
        #     "fea": 2,
        #     "hap": 3,
        #     "sad": 4,
        #     "neu": 5
        # }
        # WEIGHTED LOSS
        loss = None
        if targets is not None:
            log_probs = probs.clamp(min=1e-8).log()
            target_purity = targets.max(dim=1)[0]
            weights = torch.where(target_purity > 0.9, 2.0, 1.0)
            loss_pointwise = F.kl_div(log_probs, targets, reduction="none").sum(dim=1)
            loss = (loss_pointwise * weights).mean()

        return probs, logits, loss