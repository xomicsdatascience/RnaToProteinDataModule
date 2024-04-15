import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from RnaToProteinDataModule.BlockModules import activation, AddMRNA, AddResidual, FullyConnectedBlock, Resnet3Block, Resnet4Block, InputResidual


class NasModel(pl.LightningModule):
    def __init__(self, in_size, out_size, args):
        super().__init__()
        
        self.isAddMRNA_during = args.addMRNA_during
        self.isAddMRNA_after = args.addMRNA_after
        self.isInsertResidualAfterL1 = args.insertResidualAfterL1
        self.isInsertResidualAfterL2 = args.insertResidualAfterL2
        self.l1i_category0 = args.l1i_category0
        self.l1i_category1 = args.l1i_category1
        self.l1i_category2 = args.l1i_category2
        self.l1i_category3 = args.l1i_category3
        self.l2i_category0 = args.l2i_category0
        self.l2i_category1 = args.l2i_category1
        self.l2i_category2 = args.l2i_category2
        self.l2i_category3 = args.l2i_category3


        self.learning_rate = 0.00010348571254464918
        self.batch_size = 128
        self.in_size = in_size
        self.out_size = out_size
        self.layer1_size = 319
        self.layer2_size = 508

        self.layer1 = FullyConnectedBlock(self.in_size, c_out=self.layer1_size, act="sigmoid", dropout=0.5195300099102719, num_layers=1)
        self.category2Start = 35420
        self.category3Start = 49049

        if self.isInsertResidualAfterL1:
            self.l1iSize = 0
            if self.l1i_category0: self.l1iSize += self.out_size
            if self.l1i_category1: self.l1iSize += (self.category2Start - self.out_size)
            if self.l1i_category2: self.l1iSize += (self.category3Start - self.category2Start)
            if self.l1i_category3: self.l1iSize += (self.in_size - self.category3Start)
            self.afterLayer1_insert = InputResidual(self.layer1_size, self.l1iSize)
        else:
            self.afterLayer1_insert = AddMRNA()


        self.layer2 = Resnet3Block(self.layer1_size, self.layer2_size, act="sigmoid", dropout=0.6866931863414947)

        if self.isInsertResidualAfterL2:
            self.l2iSize = 0
            if self.l2i_category0: self.l2iSize += self.out_size
            if self.l2i_category1: self.l2iSize += (self.category2Start - self.out_size)
            if self.l2i_category2: self.l2iSize += (self.category3Start - self.category2Start)
            if self.l2i_category3: self.l2iSize += (self.in_size - self.category3Start)
            self.afterLayer2_insert = InputResidual(self.layer2_size, self.l2iSize)
        else:
            self.afterLayer2_insert = AddMRNA()

        self.layer3 = FullyConnectedBlock(self.layer2_size, c_out=self.out_size, act="tanh", dropout=0.9, num_layers=1)
        self.output_layer = nn.Linear(self.out_size, self.out_size)


    def forward(self, x):
        mRNA = x[:, :self.out_size]
        cat1Transcripts = x[:, self.out_size:self.category2Start]
        cat2Transcripts = x[:, self.category2Start:self.category3Start]
        cat3Transcripts = x[:, self.category3Start:]


        insertL1 = None
        insertL2 = None
        mRNA_during = torch.zeros_like(x[:, :self.out_size])
        mRNA_after = torch.zeros_like(x[:, :self.out_size])

        if self.isAddMRNA_during:
            mRNA_during = mRNA.clone()
        if self.isAddMRNA_after:
            mRNA_after = mRNA.clone()

        if self.isInsertResidualAfterL1:
            residuals = []
            if self.l1i_category0: residuals.append(mRNA.clone())
            if self.l1i_category1: residuals.append(cat1Transcripts.clone())
            if self.l1i_category2: residuals.append(cat2Transcripts.clone())
            if self.l1i_category3: residuals.append(cat3Transcripts.clone())
            insertL1 = torch.concatenate(residuals, axis=1)
            del residuals

        if self.isInsertResidualAfterL2:
            residuals = []
            if self.l2i_category0: residuals.append(mRNA.clone())
            if self.l2i_category1: residuals.append(cat1Transcripts.clone())
            if self.l2i_category2: residuals.append(cat2Transcripts.clone())
            if self.l2i_category3: residuals.append(cat3Transcripts.clone())
            insertL2 = torch.concatenate(residuals, axis=1)
            del residuals


        #print(mRNA_after.shape)
        #print(torch.concatenate([mRNA_after, mRNA_during]).shape)
        x = self.layer1(x)
        x = self.afterLayer1_insert(x, insertL1)
        x = self.layer2(x)
        x = self.afterLayer2_insert(x, insertL2)
        x = self.layer3(x, mRNA_during)
        x = self.output_layer(x)
        x = mRNA_after + x
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

