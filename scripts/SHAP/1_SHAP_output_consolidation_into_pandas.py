import sys
from RnaToProteinDataModule.Dataset_classes import StandardDatasetProcessor
import time
import os
import re
import numpy as np
import pandas as pd

print ('argument list', sys.argv)
proteinTarget = sys.argv[1]
save_dir = sys.argv[2]
category = save_dir.split('/')[-1]
outputDir = '1_SHAP_consolidations'
randomSeed_dataSplit = 2

curDir = os.getcwd()

dataProcessor = StandardDatasetProcessor(random_state=randomSeed_dataSplit, isOnlyCodingTranscripts=False)
dataProcessor.debug = True
dataProcessor.prepare_data()
dataProcessor.synchronize_all_datasets()
dataProcessor.split_full_dataset()
transcriptome, proteome, types = dataProcessor.extract_full_dataset()

data = []
shapFiles = os.listdir(save_dir)
proteinTargetIdx = list(proteome.columns).index(proteinTarget)
for file in sorted(shapFiles):
    print(file, flush=True)
    pattern = r'(\d)_(\d+)_(.+)_array\.npz'
    matches = re.search(pattern, file)
    if not matches: continue
    randomSeed = int(matches.group(1))
    rowIdxMatch = int(matches.group(2))
    patientId = matches.group(3)
    try:
        loaded_array = np.load(os.path.join(save_dir, file))['arr']
        proteinTargetShapValues = loaded_array[:, :, proteinTargetIdx].flatten()
        data.append([randomSeed, types[rowIdxMatch], list(transcriptome.index)[rowIdxMatch]] + list(proteinTargetShapValues))
    except Exception as e:
        print(f"Error reading file: {file} - {e}", flush=True)
        continue

df = pd.DataFrame(data, columns=['randomSeed', 'type','id']+list(transcriptome.columns))

df.to_csv(os.path.join(outputDir, f'consolidated_SHAP_{proteinTarget}_category{category}_{time.strftime("%Y%m%d-%H%M%S")}.csv'), index=False)

