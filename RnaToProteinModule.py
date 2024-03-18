import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader

from .Dataset_classes.DatasetProcessors import DatasetProcessor

class RnaToProteinDataModule(pl.LightningDataModule):
    def __init__(self, dataProcessor: DatasetProcessor):
        super().__init__()
        self.dataProcessor = dataProcessor

    def prepare_data(self):
        if hasattr(self.dataProcessor, 'X_train'): return
        self.dataProcessor.prepare_data()
        self.dataProcessor.synchronize_all_datasets()

    def setup(self, stage):
        if hasattr(self.dataProcessor, 'X_train'): return
        self.dataProcessor.split_full_dataset()
        self.input_size = self.dataProcessor.X_train.shape[1]
        self.output_size = self.dataProcessor.Y_train.shape[1]
        self.train_dataset = TensorDataset(torch.from_numpy(self.dataProcessor.X_train), torch.from_numpy(self.dataProcessor.Y_train))
        self.val_dataset = TensorDataset(torch.from_numpy(self.dataProcessor.X_val), torch.from_numpy(self.dataProcessor.Y_val))

    def train_dataloader(self):
        torch.manual_seed(self.dataProcessor.random_state)
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, shuffle=False)

