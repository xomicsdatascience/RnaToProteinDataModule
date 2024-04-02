from .smallLayers import activation, AddMRNA
from torch import nn

class FullyConnectedBlock(nn.Module):
    def __init__(self, c_in, c_out, act, dropout, num_layers, isLastBlock=False):
        super().__init__()
        before_add_layers = [
            nn.Linear(c_in, c_out),
            nn.BatchNorm1d(c_out),
            activation(act),
            nn.Dropout(p=dropout),
        ]
        if num_layers > 1:
            before_add_layers.extend([
                nn.Linear(c_out, c_out),
                nn.BatchNorm1d(c_out),
                activation(act),
                nn.Dropout(p=dropout),
            ])
        self.before_add_layers = nn.Sequential(*before_add_layers)
        self.addMRNA = AddMRNA()
        after_add_layers = [AddMRNA()]
        if num_layers > 2:
            after_add_layers.extend([
                nn.Linear(c_out, c_out),
                nn.BatchNorm1d(c_out),
                activation(act),
                nn.Dropout(p=dropout),
            ])
        self.after_add_layers = nn.Sequential(*after_add_layers)
        if isLastBlock:
            self.output_layer = nn.Linear(c_out, c_out)
        else:
            self.output_layer = AddMRNA()

    def forward(self, x, mRNA=None):
        x = self.before_add_layers(x)
        x = self.addMRNA(x, mRNA)
        x = self.after_add_layers(x)
        x = self.output_layer(x)
        return x
