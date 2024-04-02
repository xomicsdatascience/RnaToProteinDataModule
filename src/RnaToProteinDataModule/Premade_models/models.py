from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import types

from IPython.utils import io
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from RnaToProteinDataModule import RnaToProteinDataModule, NasModel

def run(model):
    if model == 'dummy': return run_dummy
    if model == 'forest': return run_forest
    if model == 'baseNN': return run_base
    if model == 'NAS14NN': return run_nas14

def run_dummy(dataProcessor):
    model = DummyRegressor()
    model.fit(dataProcessor.X_train, dataProcessor.Y_train)
    y_pred = model.predict(dataProcessor.X_val)
    mse = mean_squared_error(dataProcessor.Y_val, y_pred)
    return mse

def run_forest(dataProcessor):
    model = RandomForestRegressor(max_features='log2', max_depth=50)
    model.fit(dataProcessor.X_train, dataProcessor.Y_train)
    y_pred = model.predict(dataProcessor.X_val)
    mse = mean_squared_error(dataProcessor.Y_val, y_pred)
    return mse

def run_base(dataProcessor):
    return base_model_training(dataProcessor.X_train, dataProcessor.X_val, dataProcessor.Y_train, dataProcessor.Y_val)

def run_nas14(dataProcessor):
    args = make_args_nas14()
    dataModule = RnaToProteinDataModule(dataProcessor)
    dataModule.prepare_data()
    dataModule.setup(stage=None)
    cptac_model = NasModel(dataModule.input_size, dataModule.output_size, args)

    # Initialize a trainer (don't log anything since things get so slow...)
    trainer = Trainer(
        logger=False,
        max_epochs=5000,
        enable_progress_bar=False,
        deterministic=True,  # Do we want a bit of noise?
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)]
    )

    # Train the model and log time ⚡
    trainer.fit(model=cptac_model, datamodule=dataModule)
    with io.capture_output() as captured:
        val_loss = trainer.validate(datamodule=dataModule)[0]["val_loss"]
    return val_loss

def make_nas14(dataProcessor):
    args = make_args_nas14()
    dataModule = RnaToProteinDataModule(dataProcessor)
    dataModule.prepare_data()
    dataModule.setup(stage=None)
    cptac_model = NasModel(dataModule.input_size, dataModule.output_size, args)

    # Initialize a trainer (don't log anything since things get so slow...)
    trainer = Trainer(
        logger=False,
        deterministic=True,  # Do we want a bit of noise?
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)]
    )

    # Train the model and log time ⚡
    trainer.fit(model=cptac_model, datamodule=dataModule)
    return cptac_model, dataModule

def make_args_nas14():
    args = {}
    args["log_path"] = 'logsffzqmz3i/117'
    args["block1_exists"] = True
    args["block2_exists"] = True
    args["block3_type"] = 'fully_connect'
    args["activation3"] = 'tanh'
    args["dropout3"] = 0.9
    args["addMRNA"] = True
    args["learning_rate"] = 0.00010348571254464918
    args["batch_size"] = 128
    args["block1_type"] = 'fully_connect'
    args["hidden_size1"] = 319
    args["activation1"] = 'sigmoid'
    args["dropout1"] = 0.5195300099102719
    args["fc1"] = 1
    args["resNetType1"] = None
    args["resNetComplexConnections1"] = None
    args["block2_type"] = 'resnet'
    args["hidden_size2"] = 508
    args["activation2"] = 'sigmoid'
    args["dropout2"] = 0.6866931863414947
    args["fc2"] = None
    args["resNetType2"] = 'simple'
    args["resNetComplexConnections2"] = None
    args["fc3"] = 1
    return types.SimpleNamespace(**args)

def base_model_training(X_train, X_val, y_train, y_val):
    batch_size = 64
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    patience = 10
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], 12000)
            self.bn1 = nn.BatchNorm1d(12000)
            self.drop1 = nn.Dropout(p=0.6)
            self.fc2 = nn.Linear(12000, 10000)
            self.bn2 = nn.BatchNorm1d(10000)
            self.drop2 = nn.Dropout(p=0.6)
            self.fc3 = nn.Linear(10000, y_train.shape[1])

        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.leaky_relu(x, negative_slope=0.05)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.leaky_relu(x, negative_slope=0.05)
            x = self.drop2(x)
            x = self.fc3(x)
            return x


    class EarlyStopper:
        def __init__(self, patience=1, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_loss = float('inf')

        def early_stop(self, validation_loss):
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False

    def train(epoch, model, trainloader, optimizer, criterion):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(trainloader)
        elapsed_time = time.time() - start_time
        return epoch_loss, elapsed_time


    def validate(model, validloader, criterion):
        model.eval()
        with torch.no_grad():
            total_loss = 0.
            correct = 0.
            for data, target in validloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * len(data)
            avg_loss = total_loss / len(validloader.dataset)
            return avg_loss

    model_individual = Net().to(device)
    criterion_individual = nn.MSELoss()
    optimizer_individual = optim.Adam(model_individual.parameters())

    # Assuming X_train_normalized, y_train_normalized are numpy arrays
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                                torch.from_numpy(y_train))
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    valid_data = torch.utils.data.TensorDataset(torch.from_numpy(X_val),
                                                torch.from_numpy(y_val))
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    early_stopper = EarlyStopper(patience=patience)
    numEpochs = 5000
    for epoch in range(numEpochs):
        epoch_loss, elapsed_time = train(epoch, model_individual, trainloader, optimizer_individual,
                                         criterion_individual)
        val_loss = validate(model_individual, validloader, criterion_individual)
        if early_stopper.early_stop(val_loss) or epoch == numEpochs - 1:
            return val_loss
        return val_loss


