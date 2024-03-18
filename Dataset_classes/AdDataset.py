from .Dataset import Dataset
from .DataSplitters import StandardDataSplitter, FiveByTwoDataSplitter, NoSplitJustNormalizer
import os
import numpy as np
import pandas as pd

adDataDir = '/Users/cranneyc/Documents/Projects/CPTAC_analysis/data/AD_data'

class AdDataset(Dataset):
    def __init__(self, dataSplitter, isoformStrategy='first'):
        super().__init__(dataSplitter)
        self.load_and_preprocess_proteome()
        self.load_and_preprocess_transcriptome()
        self.match_patient_ids_between_omic_layers()
        self.deal_with_isoforms(isoformStrategy=isoformStrategy)

    def load_and_preprocess_proteome(self):
        protTmtDf = pd.read_excel(os.path.join(adDataDir, 'msbb_19batch_normalized_tmt_matrix.xlsx'), header=None)
        # labelling isoforms, removing NA or uncharacterized proteins
        geneNames = list(protTmtDf.iloc[:, 1])[5:]
        annotation = list(protTmtDf.iloc[:, 3])[5:]
        annotation = [x.lower() for x in annotation]
        protTmtDf = protTmtDf.iloc[4:, 4:]
        protTmtDf.columns = list(protTmtDf.iloc[0])
        protTmtDf = protTmtDf[1:].reset_index(drop=True)
        protTmtDf['geneNames'] = geneNames

        isoforms = []
        for i in range(len(annotation)):
            a = annotation[i]
            if 'isoform' in a or i in [689, 691, 701, 7856]:  # APOE4, APOE2, abnormal APP, abnormal PIK3R3
                isoforms.append(1)
            else:
                isoforms.append(0)

        protTmtDf['isIsoform'] = isoforms
        protTmtDf = protTmtDf.dropna(subset=['geneNames'])
        self.proteome = protTmtDf[['geneNames', 'isIsoform'] + list(protTmtDf.columns[:-2])]

    def load_and_preprocess_transcriptome(self):
        transcDf = pd.read_csv(os.path.join(adDataDir,
                                            'AMP-AD_MSBB_MSSM_BM_36.normalized.sex_race_age_RIN_PMI_exonicRate_rRnaRate_batch_adj.tsv'),
                               sep='\t')
        # identify gene names that correspond to ensemble IDs in the dataset
        ensembleDf = pd.read_csv(os.path.join(adDataDir, 'Human_ENSEMBL_Gene_ID_MSigDB.v7.0.chip'), sep='\t')
        ensembleDict = ensembleDf.set_index('Probe Set ID')['Gene Symbol'].to_dict()
        allGeneSymbolsTransc = set(transcDf['Ensembl ID'])
        allGeneSymbolsEnsembl = set(ensembleDict.keys())
        transcDf = transcDf[transcDf['Ensembl ID'].isin(allGeneSymbolsTransc.intersection(allGeneSymbolsEnsembl))]
        transcDf['geneNames'] = [ensembleDict[x] for x in transcDf['Ensembl ID']]
        self.transcriptome = transcDf[['geneNames'] + list(transcDf.columns[:-1])]

    def match_patient_ids_between_omic_layers(self):
        transcriptGeneNames = self.transcriptome['geneNames']
        transcriptEnsembleID = self.transcriptome['Ensembl ID']
        proteinGeneNames = self.proteome['geneNames']
        proteinIsIsoform = self.proteome['isIsoform']

        self.transcriptome = self.transcriptome.iloc[:, 2:]
        self.proteome = self.proteome.iloc[:, 2:]

        biospecimenDf = pd.read_csv(os.path.join(adDataDir, 'MSBB_biospecimen_metadata.csv'))
        biospecimenDf = biospecimenDf[biospecimenDf['individualID'] != 'Unknown']
        biospecimenDict = biospecimenDf.set_index('specimenID')['individualID'].to_dict()

        # change ID
        self.transcriptome.columns = [biospecimenDict[x] if x in biospecimenDict else None for x in self.transcriptome.columns]
        self.proteome.columns = [biospecimenDict[x] if x in biospecimenDict else None for x in self.proteome.columns]

        # remove unrecognized IDs
        self.transcriptome = self.transcriptome.drop(columns=[col for col in self.transcriptome.columns if col is None])
        self.proteome = self.proteome.drop(columns=[col for col in self.proteome.columns if col is None])

        # find shared IDs, remove unshared IDs
        sharedSamples = set(self.transcriptome.columns).intersection(set(self.proteome.columns))
        self.transcriptome = self.transcriptome.drop(columns=[col for col in self.transcriptome.columns if col not in sharedSamples])
        self.proteome = self.proteome.drop(columns=[col for col in self.proteome.columns if col not in sharedSamples])

        # Apparently there's a single duplicate in the proteins
        self.proteome = self.proteome.iloc[:, ~self.proteome.columns.duplicated()]
        self.transcriptome.index = pd.MultiIndex.from_arrays([transcriptGeneNames, transcriptEnsembleID], names=('geneName','Ensembl ID'))
        self.proteome.index = pd.MultiIndex.from_arrays([proteinGeneNames, proteinIsIsoform], names=('geneName', 'isIsoform'))
        self.transcriptome = self.transcriptome.T
        self.proteome = self.proteome.T

    def deal_with_isoforms(self, isoformStrategy):
        if isoformStrategy == 'first': # keeps the first/non-isoform version of a gene product
            self.proteome = self.proteome.loc[:, self.proteome.columns.get_level_values(1) != 1]
            self.proteome.columns = self.proteome.columns.droplevel(1)
            self.transcriptome = self.transcriptome.T.groupby(level='geneName').first().T

        elif isoformStrategy == 'merge': # sum all isoforms of a gene product
            self.proteome = self.proteome.T.groupby(level='geneName').sum().T
            self.transcriptome = self.transcriptome.T.groupby(level='geneName').sum().T
        self.proteome.index.name, self.transcriptome.index.name, self.transcriptome.columns.name, self.proteome.columns.name = None, None, None, None

if __name__ == "__main__":
    splitter = StandardDataSplitter()
    #splitter = NoSplitJustNormalizer()
    #splitter = FiveByTwoDataSplitter()
    ad = AdDataset(splitter)
    print(ad.transcriptome)
    print(ad.proteome)
    #d = brca.split_and_normalize(random_state=0)
    #print(d['X_train'].shape)
    #print(d['X_train_firstOrientation'].shape)