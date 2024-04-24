import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from RnaToProteinDataModule.BlockModules import activation, AddMRNA, AddResidual, FullyConnectedBlock, Resnet3Block, Resnet4Block, InputResidual


class NasModel(pl.LightningModule):
    def __init__(self, in_size, out_size, args, categoryCounts):
        super().__init__()

        self.learning_rate = 0.00010348571254464918
        self.batch_size = 128
        self.in_size = in_size
        self.out_size = out_size
        self.layer1_size = 319
        self.layer2_size = 508
        self.categoryCounts = categoryCounts
        self.category0 = args.category0
        self.category1 = args.category1
        self.category2 = args.category2
        self.category3 = args.category3
        insertSize = 0
        if self.category0: insertSize += categoryCounts[0]
        if self.category1: insertSize += categoryCounts[1]
        if self.category2: insertSize += categoryCounts[2]
        if self.category3: insertSize += categoryCounts[3]
        if insertSize == 0:
            insertSize = self.out_size

        self.layer1 = FullyConnectedBlock(self.in_size, c_out=self.layer1_size, act="sigmoid", dropout=0.5195300099102719, num_layers=1)
        self.layer2 = Resnet3Block(self.layer1_size, self.layer2_size, act="sigmoid", dropout=0.6866931863414947)
        self.layer3 = FullyConnectedBlock(self.layer2_size, c_out=insertSize, act="tanh", dropout=0.9, num_layers=1)
        self.output_layer = nn.Linear(insertSize, self.out_size)


    def forward(self, x):
        cumulative_sum = [sum(self.categoryCounts[:i + 1]) for i in range(len(self.categoryCounts))]
        mRNA = x[:, :cumulative_sum[0]]
        cat1Transcripts = x[:, cumulative_sum[0]:cumulative_sum[1]]
        cat2Transcripts = x[:, cumulative_sum[1]:cumulative_sum[2]]
        cat3Transcripts = x[:, cumulative_sum[2]:]

        residuals = []
        if self.category0: residuals.append(mRNA)
        if self.category1: residuals.append(cat1Transcripts)
        if self.category2: residuals.append(cat2Transcripts)
        if self.category3: residuals.append(cat3Transcripts)
        if len(residuals):
            residualInsert = torch.concatenate(residuals, axis=1)
        else:
            residualInsert = torch.zeros_like(mRNA)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = residualInsert + x
        x = self.output_layer(x)
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

