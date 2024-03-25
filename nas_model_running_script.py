'''
import logging
import sys
import io
from datetime import datetime

class StreamToLogger(io.TextIOBase):
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

# Configure the logging module
logging.basicConfig(filename=f'logs/output_{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}.log', level=logging.INFO)

# Redirect stdout to the logger
stdout_logger = logging.getLogger('STDOUT')
sys.stdout = StreamToLogger(stdout_logger, logging.INFO)

# Redirect stderr to the logger
stderr_logger = logging.getLogger('STDERR')
sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

print(' '.join(sys.argv))
#'''
import argparse
import logging
import os
import sys
import time
import warnings

import torch
from IPython.utils import io
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional.classification.accuracy import multiclass_accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from RnaToProteinModule import RnaToProteinDataModule
from Dataset_classes.DatasetProcessors import StandardDatasetProcessor
from NasModel import NasModel

warnings.filterwarnings("ignore")  # Disable data logger warnings
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)  # Disable GPU/TPU prints

def parse_args():
    parser = argparse.ArgumentParser(description="train mnist")
    parser.add_argument(
        "--log_path", type=str, required=True, help="dir to place tensorboard logs from all trials"
    )
    parser.add_argument("--block1_exists", action='store_true', help="exists or does not")
    parser.add_argument("--block2_exists", action='store_true', help="exists or does not")
    parser.add_argument("--block3_type", type=str, required=True, help="fully_connected, or resnet")
    parser.add_argument("--activation3", type=str, required=True, help="activation for block 3")
    parser.add_argument("--dropout3", type=float, required=True, help="dropout probability for block 3")
    parser.add_argument("--addMRNA", action='store_true', help="add mRNA to final block")
    parser.add_argument("--learning_rate", type=float, required=True, help="learning rate")
    parser.add_argument("--batch_size", type=int, required=True, help="batch size")
    parser.add_argument("--onlyCodingTranscripts", action='store_true', help="only coding transcripts used as input")
    parser.add_argument("--removeCoad", action='store_true', help="coad might by causing problems")
    parser.add_argument("--useUnsharedTranscripts", action='store_true', help="coad might by causing problems")

    parser.add_argument("--block1_type", type=str, required=False, help="fully_connected, or resnet")
    parser.add_argument("--hidden_size1", type=int, required=False, help="hidden layer size for layers in block 1")
    parser.add_argument("--activation1", type=str, required=False, help="activation for block 1")
    parser.add_argument("--dropout1", type=float, required=False, help="dropout probability for block 1")
    parser.add_argument("--fc1", type=int, required=False, help="number of layers in block 1")
    parser.add_argument("--resNetType1", type=str, required=False, help="determine type of resnet (simple vs complex)")
    parser.add_argument("--resNetComplexConnections1", type=int, required=False, help="determine connections in block 1 complex resnet")
    parser.add_argument("--block2_type", type=str, required=False, help="fully_connected, or resnet")
    parser.add_argument("--hidden_size2", type=int, required=False, help="hidden layer size for layers in block 1")
    parser.add_argument("--activation2", type=str, required=False, help="activation for block 1")
    parser.add_argument("--dropout2", type=float, required=False, help="dropout probability for block 1")
    parser.add_argument("--fc2", type=int, required=False, help="number of layers in block 1")
    parser.add_argument("--resNetType2", type=str, required=False, help="determine type of resnet (simple vs complex)")
    parser.add_argument("--resNetComplexConnections2", type=int, required=False, help="determine connections in block 1 complex resnet")
    parser.add_argument("--fc3", type=int, required=False, help="number of layers in block 3")
    return parser.parse_args()

args = parse_args()
#print(args)
epochs = 5000
#epochs = 10

def run_training_job(random_state):
    torch.manual_seed(random_state)

    dataProcessor = StandardDatasetProcessor(random_state=random_state, isOnlyCodingTranscripts=args.onlyCodingTranscripts)
    if args.removeCoad: dataProcessor.datasetNames = [x for x in dataProcessor.datasetNames if x != 'coad']
    if args.useUnsharedTranscripts: dataProcessor.isOnlyUseTranscriptsSharedBetweenDatasets = False
    dataProcessor.debug = True

    dataModule = RnaToProteinDataModule(dataProcessor)
    dataModule.prepare_data()
    dataModule.setup(stage=None)
    cptac_model = NasModel(dataModule.input_size, dataModule.output_size, args)

    # Initialize a trainer (don't log anything since things get so slow...)
    trainer = Trainer(
        logger=False,
        max_epochs=epochs,
        #enable_progress_bar=False,
        deterministic=True,  # Do we want a bit of noise?
        default_root_dir=args.log_path,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)]
    )

    # Train the model and log time ⚡
    start = time.time()
    trainer.fit(cptac_model, dataModule)
    num_epochs = trainer.current_epoch
    end = time.time()
    train_time = end - start
    print("Training completed in {} epochs.".format(num_epochs))

    # Compute the validation accuracy once and log the score
    with io.capture_output() as captured:
        val_loss = trainer.validate(datamodule=dataModule)[0]["val_loss"]
    print(f"train time: {train_time}, val loss: {val_loss}")#, num_params: {num_params}")
    return val_loss



if __name__ == "__main__":
    logger = pl_loggers.TensorBoardLogger(args.log_path)
    losses = []
    for i in range(5):
        losses.append(run_training_job(random_state=i))
    logger.log_metrics({"val_loss": np.median(losses)})
    logger.save()
