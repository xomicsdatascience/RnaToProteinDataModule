import pandas as pd
import itertools
from collections import defaultdict

pd.set_option('display.max_columns', None)
categories = pd.read_csv('data/categories_consensus.tsv', sep='\t', header=None)
categories.columns = ['gene','category']
categories['classification'] = 'other'
#categories = categories[categories['category']==1]

comprehensive = pd.read_csv('data/Human_ENSEMBL_Gene_ID_MSigDB.v7.0.chip', sep='\t')
comprehensive.drop_duplicates('Gene Symbol', keep='first', inplace=True)
proteins = pd.read_csv('data/uniprotkb_reviewed_true_AND_model_organ_2024_05_07.tsv', sep='\t')['Gene Names']
proteins = [str(x).split() for x in proteins]
proteins = set(list(itertools.chain(*proteins)))


protein_categories = categories[categories['gene'].isin(proteins)].copy()
protein_categories['classification'] = 'protein'

other_categories = categories[~categories['gene'].isin(proteins)]

comprehensive = comprehensive[comprehensive['Gene Symbol'].isin(other_categories['gene'])]

d = defaultdict(lambda: 'other')

for i, row in comprehensive.iterrows():
    if not row['Gene Title']: continue
    if 'transcription factor' in str(row['Gene Title']): d[row['Gene Symbol']] = 'transcriptionFactor'; continue
    if 'antisense' in str(row['Gene Title']): d[row['Gene Symbol']] = 'antisense'; continue
    if 'pseudogene' in str(row['Gene Title']): d[row['Gene Symbol']] = 'pseudogene'; continue
    if 'novel' in str(row['Gene Title']): d[row['Gene Symbol']] = 'novel'; continue

other_categories['classification'] = [d[x] for x in other_categories['gene']]
print(len(categories))
categories = pd.concat([protein_categories, other_categories])
print(len(categories))
categories = categories[['gene','classification']]
categories.to_csv('data/classified_categories.csv', index=False)
