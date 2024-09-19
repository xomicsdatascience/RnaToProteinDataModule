import pandas as pd
from RnaToProteinDataModule.Dataset_classes import StandardDatasetProcessor
from RnaToProteinDataModule import make_nas14_dataModule, make_nas14_model
import torch
import os
import numpy as np

randomSeed = 1
categories = '0'

torch.manual_seed(randomSeed)
dataProcessor = StandardDatasetProcessor(random_state=randomSeed, isOnlyCodingTranscripts=False)
dataModule = make_nas14_dataModule(dataProcessor)
model_save_path = f'delete_{randomSeed}.pth'

if os.path.exists(model_save_path):
    model = torch.load(model_save_path)
else:
    model = make_nas14_model(dataModule, categories)
    torch.save(model, model_save_path)

mse_per_output = []
with torch.no_grad():
    for batch in dataModule.val_dataloader(batch_size=16):
        inputs, targets = batch
        outputs = model(inputs)
        mse = (outputs - targets) ** 2
        mse_per_output.append(mse.cpu().numpy())
mse_per_output = np.concatenate(mse_per_output, axis=0)

m = mse_per_output.mean(axis=0)

transcripts, proteins, types = dataProcessor.extract_full_dataset()

mse_values_per_protein = pd.DataFrame(list(zip(m, proteins.columns)), columns=['mse', 'protein'])
mse_values_per_protein.to_csv(f'0_best_protein_decision_data/mse_per_protein_{randomSeed}.csv', index=False)
print(mse_values_per_protein)