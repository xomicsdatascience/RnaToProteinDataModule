import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from RnaToProteinDataModule.BlockModules import activation, AddMRNA, AddResidual, FullyConnectedBlock, Resnet3Block, Resnet4Block


class NasModel(pl.LightningModule):
    def __init__(self, in_size, out_size, args):
        super().__init__()

        self.isAddmRNA = args.addMRNA
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.in_size = in_size
        self.out_size = out_size
        print(in_size)
        print(out_size)
        # Set class attributes
        #self.data_dir = PATH_DATASETS

        if not args.block1_exists:
            self.layer1 = AddMRNA()
            self.layer1_outputSize = self.in_size
        elif args.block1_type == 'fully_connect':
            self.layer1 = FullyConnectedBlock(self.in_size, args.hidden_size1, args.activation1, args.dropout1, args.fc1)
            self.layer1_outputSize = args.hidden_size1
        elif args.block1_type == 'resnet' and args.resNetType1 == "simple":
            self.layer1 = Resnet3Block(self.in_size, args.hidden_size1, args.activation1, args.dropout1)
            self.layer1_outputSize = args.hidden_size1
        elif args.block1_type == 'resnet' and args.resNetType1 == "complex":
            self.layer1 = Resnet4Block(self.in_size, args.hidden_size1, args.activation1, args.dropout1, args.resNetComplexConnections1)
            self.layer1_outputSize = args.hidden_size1
        else:
            raise Exception('block 1 value error')
        if not args.block2_exists:
            self.layer2 = AddMRNA()
            self.layer2_outputSize = self.layer1_outputSize
        elif args.block2_type == 'fully_connect':
            self.layer2 = FullyConnectedBlock(self.layer1_outputSize, args.hidden_size2, args.activation2, args.dropout2, args.fc2)
            self.layer2_outputSize = args.hidden_size2
        elif args.block2_type == 'resnet' and args.resNetType2 == "simple":
            self.layer2 = Resnet3Block(self.layer1_outputSize, args.hidden_size2, args.activation2, args.dropout2)
            self.layer2_outputSize = args.hidden_size2
        elif args.block2_type == 'resnet' and args.resNetType2 == "complex":
            self.layer2 = Resnet4Block(self.layer1_outputSize, args.hidden_size2, args.activation2, args.dropout2, args.resNetComplexConnections2)
            self.layer2_outputSize = args.hidden_size2
        else:
            raise Exception('block 2 value error')

        if args.block3_type == 'fully_connect':
            self.layer3 = FullyConnectedBlock(self.layer2_outputSize, self.out_size, args.activation3, args.dropout3,
                                  args.fc3, isLastBlock=True)
        elif args.block3_type == 'resnet':
            self.layer3 = Resnet3Block(self.layer2_outputSize, self.out_size, args.activation3, args.dropout3, isLastBlock=True)
        else:
            raise Exception('block 3 value error')

    def forward(self, x):
        if self.isAddmRNA:
            mRNA = x[:, :self.out_size]
        else:
            mRNA = None
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x, mRNA)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y)
        self.log("val_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        return self(batch)

    def predict(self, input_instance):
        x = torch.from_numpy(np.array(input_instance))
        output = self(x)
        return self(x)

def get_top_10_indices(predictions, true_values):
    # Calculate Mean Squared Error (MSE) for each prediction in the batch
    mse_values = torch.mean((predictions - true_values) ** 2, dim=0)

    # Sort predictions based on MSE values
    sorted_indices = torch.argsort(mse_values)

    # Select top 10 predictions
    top_10_indices = sorted_indices[:100]
    #print('*')
    #print(len(sorted_indices))
    #print('**')

    return top_10_indices
