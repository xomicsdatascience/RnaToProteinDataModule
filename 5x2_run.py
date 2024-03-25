import pandas as pd
import numpy as np
import time
import os
import sys
import _5x2_models
from Dataset_classes.DatasetProcessors import FiveByTwoTargetDatasetProcessor

datasetNames = [
    'brca',
    'ccrcc',
    'coad',
    'gbm',
    'hnscc',
    'lscc',
    'luad',
    'ov',
    'pdac',
    # 'ucec',
    # 'ad',
    'all',
]
numDatasets = len(datasetNames)
numIterations = 100
models = ['dummy','forest','baseNN','NAS14NN']

curIdx = int(sys.argv[1])
targetModel = sys.argv[2]
targetDatasetIdx = curIdx % numDatasets
iterationNum = (curIdx // numDatasets) % numIterations
targetDataset = datasetNames[targetDatasetIdx]
print(targetDataset)
print(iterationNum)
print(targetModel)

val_loss_func = _5x2_models.run(targetModel)

for orientation in ['first','second']:
    for trainingMethod in ['allDatasets','justTargetDataset']:
        dataProcessor = FiveByTwoTargetDatasetProcessor(random_state=iterationNum, isOnlyCodingTranscripts=False, target=targetDataset, orientation=orientation, trainingMethod=trainingMethod)
        #dataProcessor.debug = True
        dataProcessor.prepare_data()
        dataProcessor.synchronize_all_datasets()
        dataProcessor.split_full_dataset()
        val_loss = val_loss_func(dataProcessor)
        print(f'final:{targetDataset},{iterationNum},{targetModel},{orientation},{trainingMethod},{val_loss}')






