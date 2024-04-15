from torch import nn

class InputResidual(nn.Module):
    def __init__(self, c_in, inputResidualSize):
        super().__init__()
        self.expand_layer = nn.Linear(c_in, inputResidualSize)
        self.condense_layer = nn.Linear(inputResidualSize, c_in)

    def forward(self, x, inputResidual):
        innerRepresentation = self.expand_layer(x)
        innerRepresentation += inputResidual
        return self.condense_layer(innerRepresentation)