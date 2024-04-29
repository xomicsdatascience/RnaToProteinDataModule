import sys
from RnaToProteinDataModule.Dataset_classes import StandardDatasetProcessor
from RnaToProteinDataModule import make_nas14_dataModule, make_nas14_model
import time
import tempfile
import os
import torch
import shap
import numpy as np

print ('argument list', sys.argv)
rowIdx = int(sys.argv[1])
randomSeed = int(sys.argv[2])
if len(sys.argv) > 3:
    categories = sys.argv[3]
else:
    categories = None

if categories:
    categoryDir = categories.replace(',', '_')
else:
    categoryDir = 'noResidual'

curDir = os.getcwd()

#save_dir = '/common/meyerjlab/caleb_SHAP_numpy_arrays_by_category'
save_dir = os.path.join(curDir, '0_SHAP_outputs')

save_dir = os.path.join(save_dir, categoryDir)

model_save_path = os.path.join(curDir, '0_SHAP_modelObjects', f'randomSeed{randomSeed}_category{categoryDir}_model.pth')
torch.manual_seed(randomSeed)

dataProcessor = StandardDatasetProcessor(random_state=randomSeed, isOnlyCodingTranscripts=False)
dataProcessor.debug = True

dataModule = make_nas14_dataModule(dataProcessor)
if os.path.exists(model_save_path):
    model = torch.load(model_save_path)
else:
    model = make_nas14_model(dataModule, categories)
    torch.save(model, model_save_path)

train_loader = dataModule.train_dataloader()
background, _ = next(iter(train_loader))
transcriptome, _, _ = dataProcessor.extract_full_dataset()

test_images = torch.from_numpy(transcriptome.to_numpy()).to(torch.float32)
outputFile = f'{randomSeed}_{str(rowIdx).zfill(len(str(len(transcriptome))))}_{transcriptome.index[rowIdx]}_array.npz'
print(outputFile)

e = shap.DeepExplainer(model, background)

shap_values = e.shap_values(test_images[rowIdx:rowIdx+1], check_additivity=False)
shap_values = np.array(shap_values)
print(shap_values)
# Save the array to file
np.savez_compressed(os.path.join(save_dir, outputFile), arr=shap_values)
#loaded_array = np.load(file_path)['arr']

