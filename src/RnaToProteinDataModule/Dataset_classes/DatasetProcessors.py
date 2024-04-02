from .CptacDataset import CptacDataset
from .AdDataset import AdDataset
from .DataSplitters import StandardDataSplitter, FiveByTwoDataSplitter, NoSplitJustNormalizer
from abc import ABC, abstractmethod
import random
import numpy as np
from collections import OrderedDict
import pandas as pd

class DatasetProcessor(ABC):
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
        # 'ucec',
        # 'ad',
    ]
    datasets = OrderedDict()
    random_state = 0
    debug = False
    isOnlyUseTranscriptsSharedBetweenDatasets = True

    def __init__(self, random_state, isOnlyCodingTranscripts):
        self.random_state = random_state
        self.isOnlyCodingTranscripts = isOnlyCodingTranscripts


    def synchronize_all_datasets(self):

        self.allProteinGeneTargets = self.identify_all_shared_targets('proteome')
        self.allTranscriptGeneTargets = self.identify_transcript_targets()

        self.ensure_mrna_direct_precursors_to_proteins_listed_first_in_transcriptome()

        if self.debug:
            self.allProteinGeneTargets = self.allProteinGeneTargets[:100]
            self.allTranscriptGeneTargets = self.allTranscriptGeneTargets[:500]

        if self.isOnlyCodingTranscripts:
            self.allTranscriptGeneTargets = self.allProteinGeneTargets.copy()

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
        if self.isOnlyUseTranscriptsSharedBetweenDatasets:
            return self.identify_all_shared_targets('transcriptome')
        else:
            return self.identify_all_targets('transcriptome')

    def filter_transcripts_to_targets(self, targetTranscripts, targetProtein):
        targetProteinIdx = self.allTranscriptGeneTargets.index(targetProtein)
        targetProteinColumnTrain = self.X_train[:, targetProteinIdx]
        targetProteinColumnVal = self.X_val[:, targetProteinIdx]
        self.X_train = np.delete(self.X_train, targetProteinIdx, axis=1)
        self.X_val = np.delete(self.X_val, targetProteinIdx, axis=1)
        self.allTranscriptGeneTargets.pop(targetProteinIdx)

        indices_to_keep = [self.allTranscriptGeneTargets.index(target) for target in targetTranscripts]
        self.X_train = self.X_train[:, indices_to_keep]
        self.X_val = self.X_val[:, indices_to_keep]
        self.allTranscriptGeneTargets = [transcript for transcript in self.allTranscriptGeneTargets if transcript in targetTranscripts]

        self.X_train = np.insert(self.X_train, 0, targetProteinColumnTrain, axis=1)
        self.X_val = np.insert(self.X_val, 0, targetProteinColumnVal, axis=1)
        self.allTranscriptGeneTargets.insert(0, targetProtein)

    def filter_to_target_protein(self, targetProtein):
        index = self.allProteinGeneTargets.index(targetProtein)
        self.Y_train = self.Y_train[:, index].reshape(-1, 1)
        self.Y_val = self.Y_val[:, index].reshape(-1, 1)
        self.allProteinGeneTargets = [targetProtein]




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

    def extract_full_dataset(self):
        X = []
        Y = []
        for datasetName, dataset in self.datasets.items():
            X.append(dataset.transcriptome)
            Y.append(dataset.proteome)
        return pd.concat(X), pd.concat(Y)

    def ensure_mrna_direct_precursors_to_proteins_listed_first_in_transcriptome(self):
        mRNAs_with_direct_protein_match = set(self.allTranscriptGeneTargets).intersection(self.allProteinGeneTargets)
        mRNAs_without_direct_protein_match = set(self.allTranscriptGeneTargets) - set(mRNAs_with_direct_protein_match)
        self.allProteinGeneTargets = sorted(mRNAs_with_direct_protein_match)
        self.allTranscriptGeneTargets = sorted(mRNAs_with_direct_protein_match) + sorted(mRNAs_without_direct_protein_match)


    def prepare_data(self):
        self.datasets = {}
        for datasetName in self.datasetNames:
            datasetSplitter = self.return_data_splitter(datasetName)
            if datasetSplitter == None: continue
            datasetSplitter.random_state = self.random_state
            self.datasets[datasetName] = return_dataset(datasetSplitter, datasetName)

    @abstractmethod
    def return_data_splitter(self, datasetName):
        pass

class FiveByTwoTargetDatasetProcessor(DatasetProcessor):
    def __init__(self, random_state, isOnlyCodingTranscripts, target, orientation, trainingMethod):
        super().__init__(random_state=random_state, isOnlyCodingTranscripts=isOnlyCodingTranscripts)
        self.target = target
        self.orientation = orientation
        self.trainingMethod = trainingMethod

    def return_data_splitter(self, datasetName):
        if self.target == datasetName or self.target == 'all':
            return FiveByTwoDataSplitter(random_state=self.random_state, orientation=self.orientation)
        elif self.trainingMethod == 'allDatasets':
            return NoSplitJustNormalizer()
        elif self.trainingMethod == 'justTargetDataset':
            return

class TargetDatasetProcessor(DatasetProcessor):
    def __init__(self, random_state, target):
        super().__init__(random_state=random_state)
        self.target = target

    def return_data_splitter(self, datasetName):
        if self.target == datasetName:
            dataSplitter = StandardDataSplitter(random_state=self.random_state, val_size=0.2)
            return dataSplitter
        else:
            return NoSplitJustNormalizer(random_state=self.random_state)

class StandardDatasetProcessor(DatasetProcessor):
    def return_data_splitter(self, datasetName):
        return StandardDataSplitter(random_state=self.random_state)



def return_dataset(datasetSplitter, datasetName):
    if datasetName == 'ad':
        dataset = AdDataset(datasetSplitter)
    else:
        dataset = CptacDataset(datasetSplitter, datasetName)
    return dataset