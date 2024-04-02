import sys
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from RnaToProteinDataModule.Dataset_classes import StandardDatasetProcessor
from RnaToProteinDataModule import make_nas14
import time
import tempfile
import os
import torch
import shap
import numpy as np

print ('argument list', sys.argv)
rowIdx = int(sys.argv[1])
randomSeed = int(sys.argv[2])
randomSeed_dataSplit = 2

#save_dir = '/common/meyerjlab/caleb_SHAP_numpy_arrays'
save_dir = '/Users/cranneyc/Documents/Projects/CPTAC_analysis/makingABetterModel_NAS/RnaToProteinDataModule/scripts/SHAP/outputs'
curDir = os.getcwd()
log_dir = tempfile.mkdtemp(prefix='logs_zDELETE', dir=curDir)
torch.manual_seed(randomSeed)

dataProcessor = StandardDatasetProcessor(random_state=randomSeed_dataSplit, isOnlyCodingTranscripts=False)
dataProcessor.debug = True
dataProcessor.prepare_data()
dataProcessor.synchronize_all_datasets()
dataProcessor.split_full_dataset()

model, dataModule = make_nas14(dataProcessor)

train_loader = dataModule.train_dataloader()
background, _ = next(iter(train_loader))
transcriptome, _ = dataProcessor.extract_full_dataset()

test_images = torch.from_numpy(transcriptome.to_numpy()).to(torch.float32)
outputFile = f'{randomSeed}_{str(rowIdx).zfill(len(str(len(transcriptome))))}_{transcriptome.index[rowIdx]}_array.npz'
print(outputFile)

e = shap.DeepExplainer(model, background)

shap_values = e.shap_values(test_images[rowIdx:rowIdx+1], check_additivity=False)
shap_values = np.array(shap_values)

# Save the array to file
np.savez_compressed(os.path.join(save_dir, outputFile), arr=shap_values)
#loaded_array = np.load(file_path)['arr']

