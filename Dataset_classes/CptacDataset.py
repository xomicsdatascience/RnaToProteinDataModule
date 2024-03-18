import cptac
from .Dataset import Dataset
from .DataSplitters import StandardDataSplitter, FiveByTwoDataSplitter, NoSplitJustNormalizer

class CptacDataset(Dataset):
    def __init__(self, dataSplitter, type, source='bcm', isoformStrategy='first'):
        super().__init__(dataSplitter)
        self.cptacMod  = get_cptac_mod(type)
        self.source = source
        self.proteome = self.cptacMod.get_proteomics(source=self.source).fillna(0)
        self.transcriptome = self.cptacMod.get_transcriptomics(source=self.source).fillna(0)
        self.match_patient_ids_between_omic_layers()
        self.deal_with_isoforms(isoformStrategy)

    def match_patient_ids_between_omic_layers(self):
        common_indices = self.proteome.index.intersection(self.transcriptome.index)
        self.proteome = self.proteome.loc[common_indices]
        self.transcriptome = self.transcriptome.loc[common_indices]

    def deal_with_isoforms(self, isoformStrategy):
        if isoformStrategy == 'first': # keeps first instance of a gene product
            self.proteome = self.proteome.T.groupby(level='Name').first().T
            self.transcriptome = self.transcriptome.T.groupby(level='Name').first().T

        elif isoformStrategy == 'merge': # sum all isoforms of a gene product
            self.proteome = self.proteome.T.groupby(level='Name').sum().T
            self.transcriptome = self.transcriptome.T.groupby(level='Name').sum().T
        self.proteome.index.name, self.transcriptome.index.name, self.transcriptome.columns.name, self.proteome.columns.name = None, None, None, None

def get_cptac_mod(cancerType):
    if cancerType == 'brca':
        return cptac.Brca()
    elif cancerType == 'ccrcc':
        return cptac.Ccrcc()
    elif cancerType == 'coad':
        return cptac.Coad()
    elif cancerType == 'gbm':
        return cptac.Gbm()
    elif cancerType == 'hnscc':
        return cptac.Hnscc()
    elif cancerType == 'lscc':
        return cptac.Lscc()
    elif cancerType == 'luad':
        return cptac.Luad()
    elif cancerType == 'ov':
        return cptac.Ov()
    elif cancerType == 'pdac':
        return cptac.Pdac()
    elif cancerType == 'ucec':
        return cptac.Ucec()
    else:
        raise Exception('cancer type not found in CPTAC datasets')

if __name__ == "__main__":
    #splitter = StandardDataSplitter()
    #splitter = NoSplitJustNormalizer()
    splitter = FiveByTwoDataSplitter()
    brca = CptacDataset(splitter, type='brca')
    print(brca.transcriptome)
    d = brca.split_and_normalize(random_state=0)
    #print(d['X_train'].shape)
    print(d['X_train_firstOrientation'].shape)