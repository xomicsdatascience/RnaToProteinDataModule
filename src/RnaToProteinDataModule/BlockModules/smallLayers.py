from torch import nn

def activation(activation_param):
    if activation_param == 'leaky_relu_steep':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation_param == 'leaky_relu_slight':
        return nn.LeakyReLU(negative_slope=0.01)
    elif activation_param == 'sigmoid':
        return nn.Sigmoid()
    elif activation_param == 'tanh':
        return nn.Tanh()
    elif activation_param == 'rrelu':
        return nn.RReLU()
    elif activation_param == 'selu':
        return nn.SELU()


class AddMRNA(nn.Module):
    def forward(self, x, mRNA=None):
        if mRNA is not None:
            return x + mRNA
        else:
            return x

class AddResidual(nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.isAddResidual = residual
        #self.isAddResidual)
    def forward(self, x, y):
        if self.isAddResidual:
            return x + y
        else:
            return x
    def __str__(self):
        return f'AddResidual: {self.isAddResidual}'
