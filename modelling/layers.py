import numpy as np
import torch
import torch.nn.functional as F

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim:int, hidden_layers_dim:list=[32,16,8], dropout:float=0.2, output_layer:bool=True):
        """Multilayer perceptron object

        Args:
            input_dim (int): Input layer dimension
            hidden_layers_dim (list, optional): Hidden layers dimensions. Defaults to [32,16,8].
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            output_layer (bool, optional): Add output layer with dimension 1. Defaults to True.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers_dim = hidden_layers_dim
        layers = list()
        for out_dim in hidden_layers_dim:  
            layers.append(torch.nn.Linear(input_dim, out_dim))
            layers.append(torch.nn.BatchNorm1d(out_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0.0:
              layers.append(torch.nn.Dropout(p=dropout))
            input_dim = out_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x:torch.Tensor):
        """Forward method

        Args:
            x (torch.Tensor): Float tensor of size ``(batch_size, embed_dim)``

        Returns:
            _type_: _description_
        """
        return self.mlp(x)

class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias