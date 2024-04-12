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

#'''
absMeanShapDict, meanShapDict = make_dict_of_absolute_mean_SHAP(file)
spearmanDict = make_dict_of_spearman_values(protein)
shared_keys = set(absMeanShapDict.keys()) & set(spearmanDict.keys())
merged_dict = {key: {'absMeanSHAP': absMeanShapDict[key], 'spearman': spearmanDict[key], 'meanSHAP': meanShapDict[key]} for key in shared_keys}
df = pd.DataFrame.from_dict(merged_dict, orient='index')

df.to_csv(os.path.join(outputDir, f'{protein}_shap_vs_spearman.csv'))
#'''
#df = pd.read_csv(os.path.join(outputDir, f'{protein}_shap_vs_spearman.csv'), index_col=0)

t, p = get_transcriptome_proteome()
tc, pc = t.columns, p.columns


print(len(df[df['meanSHAP'] > 0]))
print(len(df[df['meanSHAP'] < 0]))
print(len(df[df['meanSHAP'] == 0]))
print('*')

arr = np.log(df['absMeanSHAP'])
shapArr = np.array(df['absMeanSHAP'])
indexArr = np.array(df.index)
arr[np.isinf(arr)] = -20
arr[np.isneginf(arr)] = -20
df['absMeanSHAP'] = arr

transcripts = list(t.columns)
numCoding = len(p.columns)
df['order'] = df.index.map(lambda x: transcripts.index(x) if x in transcripts else -1)

print(len(df[(df['absMeanSHAP'] > -10) & (df['order'] >= numCoding)]))
print(len(df[(df['absMeanSHAP'] > -10) & (df['order'] < numCoding)]))
print(len(df[(df['absMeanSHAP'] <= -10) & (df['order'] >= numCoding)]))
print(len(df[(df['absMeanSHAP'] <= -10) & (df['order'] < numCoding)]))
print(df[(df['absMeanSHAP'] <= -10) & (df['order'] <= numCoding)])
print(len(set(df[(df['absMeanSHAP'] <= -10) & (df['order'] < numCoding)].index).intersection(set(p.columns))))
print(set(df[(df['absMeanSHAP'] <= -10) & (df['order'] < numCoding)].index) - set(p.columns))
print(df)



#'''
#colors = [(1, 1, 1)] + sns.color_palette("YlGnBu", as_cmap=True)
#custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)


# Plain scatterplot
plt.figure(figsize=(8, 6))
#sns.scatterplot(data=df, x='spearman', y='absMeanSHAP', cmap='YlGnBu', s=100)
plot = sns.jointplot(data=df, x='spearman', y='absMeanSHAP', kind='hex', cmap="Blues", vmax=250)
#plt.scatter(df['spearman'], df['absMeanSHAP'])
#plt.hexbin(df['spearman'], df['absMeanSHAP'], gridsize=50, cmap='YlGnBu', mincnt=1)

plt.xlabel('spearman')
plt.ylabel('absMeanSHAP (log)')
#plt.yscale('log')
#plt.ylim(-25, 0)
#plt.title('Horseshoe Plot')
#plt.grid(True)
plt.savefig(os.path.join(outputDir, f'{protein}_horseshoe.svg'))
#'''

'''
pivot_table = df.pivot_table(index='absMeanSHAP', columns='spearman', aggfunc='size', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='g', cbar_kws={'label': 'Density'})
plt.xlabel('Value from dict1')
plt.ylabel('Value from dict2')
plt.title('Heatmap of Density')
plt.show()
'''