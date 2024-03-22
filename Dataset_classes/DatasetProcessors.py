from .CptacDataset import CptacDataset
from .AdDataset import AdDataset
from .DataSplitters import StandardDataSplitter, FiveByTwoDataSplitter, NoSplitJustNormalizer
from abc import ABC, abstractmethod
import random
import numpy as np

datasetNames = [
    'brca',
    'ccrcc',
    'coad',
    'gbm',
    'hnscc',
    'lscc',
    'luad',
    'ov',
    'pdac',
    #'ucec',
    #'ad',
]

class DatasetProcessor(ABC):
    datasets = dict()
    random_state = 0
    debug = False
    isTranscriptOnlyShared = True

    def synchronize_all_datasets(self):

        self.allProteinGeneTargets = self.identify_all_shared_targets('proteome')
        self.allTranscriptGeneTargets = self.identify_transcript_targets()

        self.ensure_mrna_direct_precursors_to_proteins_listed_first_in_transcriptome()

        if self.debug:
            self.allProteinGeneTargets = self.allProteinGeneTargets[:100]
            self.allTranscriptGeneTargets = self.allTranscriptGeneTargets[:500]

        # only use common proteins/transcripts
        for datasetName, dataset in self.datasets.items():
            self.datasets[datasetName].filter_to_only_include_given_genes('proteome', self.allProteinGeneTargets)
            self.datasets[datasetName].filter_to_only_include_given_genes('transcriptome', self.allTranscriptGeneTargets)

        #del self.datasets['ad']

    def identify_all_shared_targets(self, omicLayer):
        random.seed(self.random_state)
        randomDatasetName = random.choice(list(self.datasets.keys()))
        sharedTargets = set(self.datasets[randomDatasetName].get_gene_names(omicLayer))
        for datasetName, dataset in self.datasets.items():
            if datasetName == randomDatasetName: continue
            sharedTargets = sharedTargets.intersection(dataset.get_gene_names(omicLayer))
        return sorted(sharedTargets)

    def identify_all_targets(self, omicLayer):
        random.seed(self.random_state)
        randomDatasetName = random.choice(list(self.datasets.keys()))
        allTargets = set(self.datasets[randomDatasetName].get_gene_names(omicLayer))
        for datasetName, dataset in self.datasets.items():
            if datasetName == randomDatasetName: continue
            allTargets = allTargets | set(dataset.get_gene_names(omicLayer))
        return sorted(allTargets)

    def identify_transcript_targets(self):
        if self.isTranscriptOnlyShared:
            return self.identify_all_shared_targets('transcriptome')
        else:
            return self.identify_all_targets('transcriptome')


    def split_full_dataset(self):
        X_train = []
        X_val = []
        Y_train = []
        Y_val = []
        for datasetName, dataset in self.datasets.items():
            data = dataset.split_and_normalize()
            X_train.append(data['X_train'])
            if 'X_val' in data: X_val.append(data['X_val'])
            Y_train.append(data['Y_train'])
            if 'Y_val' in data: Y_val.append(data['Y_val'])
        self.X_train = np.concatenate(X_train)
        self.X_val = np.concatenate(X_val)
        self.Y_train = np.concatenate(Y_train)
        self.Y_val = np.concatenate(Y_val)

    def ensure_mrna_direct_precursors_to_proteins_listed_first_in_transcriptome(self):
        mRNAs_with_direct_protein_match = set(self.allTranscriptGeneTargets).intersection(self.allProteinGeneTargets)
        mRNAs_without_direct_protein_match = set(self.allTranscriptGeneTargets) - set(mRNAs_with_direct_protein_match)
        self.allProteinGeneTargets = sorted(mRNAs_with_direct_protein_match)
        self.allTranscriptGeneTargets = sorted(mRNAs_with_direct_protein_match) + sorted(mRNAs_without_direct_protein_match)


    def prepare_data(self):
        self.datasets = {}
        for datasetName in datasetNames:
            datasetSplitter = self.return_data_splitter(datasetName)
            datasetSplitter.random_state = self.random_state
            self.datasets[datasetName] = return_dataset(datasetSplitter, datasetName)

    @abstractmethod
    def return_data_splitter(self, datasetName):
        pass

class FiveByTwoTargetDatasetProcessor(DatasetProcessor):
    def __init__(self, random_state, target, orientation):
        self.random_state = random_state
        self.target = target
        self.orientation = orientation

    def return_data_splitter(self, datasetName):
        if self.target == datasetName or self.target == 'all':
            return FiveByTwoDataSplitter(self.orientation)
        else:
            return NoSplitJustNormalizer()

class TargetDatasetProcessor(DatasetProcessor):
    def __init__(self, random_state, target):
        self.random_state = random_state
        self.target = target

    def return_data_splitter(self, datasetName):
        if self.target == datasetName:
            dataSplitter = StandardDataSplitter()
            dataSplitter.val_size = 0.2
            return dataSplitter
        else:
            return NoSplitJustNormalizer()

class StandardDatasetProcessor(DatasetProcessor):
    def __init__(self, random_state=0):
        self.random_state = random_state

    def return_data_splitter(self, datasetName):
        return StandardDataSplitter()



def return_dataset(datasetSplitter, datasetName):
    if datasetName == 'ad':
        dataset = AdDataset(datasetSplitter)
    else:
        dataset = CptacDataset(datasetSplitter, datasetName)
    return dataset