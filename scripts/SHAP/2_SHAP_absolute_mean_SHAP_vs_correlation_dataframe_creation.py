import pandas as pd
import numpy as np
import torch
from RnaToProteinDataModule.Dataset_classes import StandardDatasetProcessor
from RnaToProteinDataModule import make_nas14
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import sys
import re
import os

file = sys.argv[1]
outputDir = sys.argv[2]

print(file)
randomSeed = 1

def make_dict_of_absolute_mean_SHAP(file):
    df = pd.read_csv(file)
    df = df[df['randomSeed']==randomSeed]
    df.drop(columns=['id', 'randomSeed', 'type'], inplace=True)
    abs_mean_values = df.abs().mean()
    mean_values = df.mean()
    return abs_mean_values.to_dict(), mean_values.to_dict()

def get_transcriptome_proteome():
    randomSeed_dataSplit = 2
    torch.manual_seed(randomSeed)

    dataProcessor = StandardDatasetProcessor(random_state=randomSeed_dataSplit, isOnlyCodingTranscripts=False)
    #dataProcessor.debug = True
    dataProcessor.prepare_data()
    dataProcessor.synchronize_all_datasets()
    dataProcessor.split_full_dataset()

    transcriptome, proteome, _ = dataProcessor.extract_full_dataset()
    return transcriptome, proteome

def make_dict_of_spearman_values(protein):
    transcriptome, proteome = get_transcriptome_proteome()
    column_to_correlate_with = protein
    correlation_results = transcriptome.apply(lambda x: proteome[column_to_correlate_with].corr(x, method='spearman'))
    return correlation_results.to_dict()

pattern = r"consolidated_SHAP_(\w+)_\d+"
match = re.search(pattern, file)
protein = match.group(1)

absMeanShapDict, meanShapDict = make_dict_of_absolute_mean_SHAP(file)
spearmanDict = make_dict_of_spearman_values(protein)
shared_keys = set(absMeanShapDict.keys()) & set(spearmanDict.keys())
merged_dict = {key: {'absMeanSHAP': absMeanShapDict[key], 'spearman': spearmanDict[key], 'meanSHAP': meanShapDict[key]} for key in shared_keys}
df = pd.DataFrame.from_dict(merged_dict, orient='index')

df.to_csv(os.path.join(outputDir, f'{protein}_shap_vs_spearman.csv'))
