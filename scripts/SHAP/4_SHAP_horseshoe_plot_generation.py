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

absShapCorrelationFile = sys.argv[1]
categoryConsensusFilePath = sys.argv[2]
outputDir = sys.argv[3]

print(absShapCorrelationFile)

df = pd.read_csv(absShapCorrelationFile, index_col=0)

arr = np.log(df['absMeanSHAP'])
shapArr = np.array(df['absMeanSHAP'])
indexArr = np.array(df.index)
lowest_non_inf = np.min(arr[np.isfinite(arr)])
arr[np.isinf(arr)] = lowest_non_inf
arr[np.isneginf(arr)] = lowest_non_inf
df['absMeanSHAP'] = arr

categoryDict = dict(np.genfromtxt(categoryConsensusFilePath, delimiter='\t', dtype=str))
df['category'] = [int(categoryDict[x]) for x in df.index]
df = df[df['category'] != 3]

plt.figure(figsize=(8, 6))
#sns.scatterplot(data=df, x='spearman', y='absMeanSHAP', cmap='YlGnBu', s=0.75)
#plot = sns.jointplot(data=df, x='spearman', y='absMeanSHAP', kind='hex', cmap="Blues", vmax=250)
plot = sns.jointplot(data=df, x='spearman', y='absMeanSHAP', hue='category', kind='kde', palette={0: 'blue', 1: 'red', 2:'green', 3:'orange'})

plt.xlabel('spearman')
plt.ylabel('absMeanSHAP (log)')

pattern = r"(\w+)_shap_vs_spearman"
match = re.search(pattern, absShapCorrelationFile)
protein = match.group(1)

plt.savefig(os.path.join(outputDir, f'{protein}_horseshoe_experiment.svg'))
#plt.show()
