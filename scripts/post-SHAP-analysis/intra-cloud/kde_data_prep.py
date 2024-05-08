import pandas as pd
import numpy as np
import sys
import os

absShapCorrelationFile = sys.argv[1]
categoryConsensusFilePath = sys.argv[2]

df = pd.read_csv(absShapCorrelationFile, index_col=0)

arr = np.log(df['absMeanSHAP'])
shapArr = np.array(df['absMeanSHAP'])
indexArr = np.array(df.index)
lowest_non_inf = np.min(arr[np.isfinite(arr)])
arr[np.isinf(arr)] = lowest_non_inf
arr[np.isneginf(arr)] = lowest_non_inf
df['absMeanSHAP'] = arr

geneDf = pd.read_csv(categoryConsensusFilePath)

classificationDict = geneDf.set_index('gene').to_dict()['classification']
categoryDict = geneDf.set_index('gene').to_dict()['category']
df['classification'] = [classificationDict[x] for x in df.index]
df['category'] = [categoryDict[x] for x in df.index]

df = df[df['category']==1]
#HACK
df['category'] = df['classification']

with open('data.json', 'w') as f:
    f.write(df.to_json(orient='records'))


