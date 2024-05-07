from .smallLayers import activation, AddMRNA
from torch import nn

class Resnet3Block(nn.Module):
    def __init__(self, c_in, c_out, act, dropout, isLastBlock=False):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.BatchNorm1d(c_out),
            activation(act),
            nn.Dropout(p=dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(c_out, c_out),
            nn.BatchNorm1d(c_out),
            activation(act),
            nn.Dropout(p=dropout),
        )
        self.addMRNA = AddMRNA()
        self.layer3 = nn.Sequential(
            nn.Linear(c_out, c_out),
            nn.BatchNorm1d(c_out),
        )
        self.final_act = activation(act)
        self.final_dropout = nn.Dropout(p=dropout)
        if isLastBlock:
            self.output_layer = nn.Linear(c_out, c_out)
        else:
            self.output_layer = AddMRNA()


    def forward(self, x, mRNA=None):
        if mRNA is not None:
            pass
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y2 = self.addMRNA(y2, y1)
        y3 = self.layer3(y2)
        final_y = y3 + y1
        final_y = self.final_act(final_y)
        final_y = self.final_dropout(final_y)
        final_y = self.output_layer(final_y)
        return final_y
