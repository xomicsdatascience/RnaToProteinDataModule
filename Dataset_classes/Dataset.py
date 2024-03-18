from abc import ABC, abstractmethod
from .DataSplitters import DataSplitter

class Dataset(ABC):

    proteome = None
    transcriptome = None

    def __init__(self, dataSplitter: DataSplitter):
        self.dataSplitter = dataSplitter

    @abstractmethod
    def match_patient_ids_between_omic_layers(self):
        pass

    @abstractmethod
    def deal_with_isoforms(self, isoformStrategy):
        pass

    def get_gene_names(self, varName):
        target = self.return_target_dataset(varName)
        return list(target.columns)

    def filter_to_only_include_given_genes(self, varName, genes):
        target = self.return_target_dataset(varName)
        target = reorder_columns(target, genes)
        setattr(self, varName, target)

    def return_target_dataset(self, varName):
        return getattr(self, varName)

    def split_and_normalize(self):
        return self.dataSplitter(X=self.transcriptome, Y=self.proteome)


def reorder_columns(df, column_list):
    existing_columns = set(df.columns)
    new_columns = set(column_list) - existing_columns

    for col in new_columns:
        df[col] = 0

    df = df.reindex(columns=column_list, fill_value=0)

    return df
