from .smallLayers import activation, AddMRNA, AddResidual
from torch import nn

def decimal_to_binary_array(decimal_number, array_length):
    binary_string = format(decimal_number, f'0{array_length}b')
    binary_array = [int(bit) for bit in binary_string]
    return binary_array

class Resnet4Block(nn.Module):
    def __init__(self, c_in, c_out, act, dropout, resnetConnectionInt, isLastBlock=False):
        super().__init__()
        resnetConnections = decimal_to_binary_array(resnetConnectionInt, 5)
        self.layer1 = nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.BatchNorm1d(c_out),
            activation(act),
            nn.Dropout(p=dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(c_out, c_out),
            nn.BatchNorm1d(c_out),
        )
        self.layer1_layer2_residual = AddResidual(resnetConnections[0])
        #print(self.layer1_layer2_residual)
        self.layer2_adjust = nn.Sequential(
            activation(act),
            nn.Dropout(p=dropout),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(c_out, c_out),
            nn.BatchNorm1d(c_out),
        )
        self.layer1_layer3_residual = AddResidual(resnetConnections[1])
        self.layer2_layer3_residual = AddResidual(resnetConnections[2])
        #print(self.layer1_layer3_residual)
        #print(self.layer2_layer3_residual)
        self.layer3_adjust = nn.Sequential(
            activation(act),
            nn.Dropout(p=dropout),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(c_out, c_out),
            nn.BatchNorm1d(c_out),
        )
        self.layer1_layer4_residual = AddResidual(resnetConnections[3])
        self.layer2_layer4_residual = AddResidual(resnetConnections[4])
        #print(self.layer1_layer4_residual)
        #print(self.layer2_layer4_residual)
        self.layer4_adjust = nn.Sequential(
            activation(act),
            nn.Dropout(p=dropout),
        )
        if isLastBlock:
            self.output_layer = nn.Linear(c_out, c_out)
        else:
            self.output_layer = AddMRNA()


    def forward(self, x):
        y1 = self.layer1(x)
        pre_y2 = self.layer2(y1)
        pre_y2 = self.layer1_layer2_residual(y1, pre_y2)
        y2 = self.layer2_adjust(pre_y2)
        pre_y3 = self.layer3(y2)
        pre_y3 = self.layer1_layer3_residual(y1, pre_y3)
        pre_y3 = self.layer2_layer3_residual(y2, pre_y3)
        y3 = self.layer2_adjust(pre_y3)
        pre_y4 = self.layer4(y3)
        pre_y4 = self.layer1_layer4_residual(y1, pre_y4)
        pre_y4 = self.layer2_layer4_residual(y2, pre_y4)
        y4 = self.layer4_adjust(pre_y4)
        y4 = self.output_layer(y4)
        return y4
