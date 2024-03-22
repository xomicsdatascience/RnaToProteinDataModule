from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod

class DataSplitter(ABC):
    random_state = 0

    @abstractmethod
    def split_and_normalize(self, X, Y):
        pass

    def __call__(self, *args, **kwargs):
        return self.split_and_normalize(kwargs['X'], kwargs['Y'])


class StandardDataSplitter(DataSplitter):
    val_size = 0.1

    def split_and_normalize(self, X, Y):
        scalerX = StandardScaler()
        scalerY = StandardScaler()
        X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, test_size=self.val_size * 2,
                                                                    random_state=self.random_state)
        X_val, _, Y_val, _ = train_test_split(X_val_test, Y_val_test, test_size=0.5,
                                                                    random_state=self.random_state)
        X_train = scalerX.fit_transform(X_train).astype(np.float32)
        X_val = scalerX.transform(X_val).astype(np.float32)
        Y_train = scalerY.fit_transform(Y_train).astype(np.float32)
        Y_val = scalerY.transform(Y_val).astype(np.float32)
        return {
            'X_train': X_train,
            'X_val': X_val,
            'Y_train': Y_train,
            'Y_val': Y_val,
        }


class NoSplitJustNormalizer(DataSplitter):
    def split_and_normalize(self, X, Y):
        scalerX = StandardScaler()
        scalerY = StandardScaler()
        X_train = scalerX.fit_transform(X).astype(np.float32)
        Y_train = scalerY.fit_transform(Y).astype(np.float32)
        return {
            'X_train': X_train,
            'Y_train': Y_train,
        }


class FiveByTwoDataSplitter(DataSplitter):
    test_size = 0.1

    def __init__(self, orientation):
        self.orientation = orientation

    def split_and_normalize(self, X, Y):
        scalerX = StandardScaler()
        scalerY = StandardScaler()
        X_one_two, _, Y_one_two, _ = train_test_split(X, Y, test_size=self.test_size,
                                                                    random_state=self.random_state)
        X_one, X_two, Y_one, Y_two = train_test_split(X_one_two, Y_one_two, test_size=0.5,
                                                                    random_state=self.random_state)

        if self.orientation == 'first':
            X_train_firstOrientation = scalerX.fit_transform(X_one).astype(np.float32)
            X_val_firstOrientation = scalerX.transform(X_two).astype(np.float32)
            Y_train_firstOrientation = scalerY.fit_transform(Y_one).astype(np.float32)
            Y_val_firstOrientation = scalerY.transform(Y_two).astype(np.float32)

            return {
                'X_train': X_train_firstOrientation,
                'X_val': X_val_firstOrientation,
                'Y_train': Y_train_firstOrientation,
                'Y_val': Y_val_firstOrientation,
            }
        else:
            X_train_secondOrientation = scalerX.fit_transform(X_two).astype(np.float32)
            X_val_secondOrientation = scalerX.transform(X_one).astype(np.float32)
            Y_train_secondOrientation = scalerY.fit_transform(Y_two).astype(np.float32)
            Y_val_secondOrientation = scalerY.transform(Y_one).astype(np.float32)

            return {
                'X_train': X_train_secondOrientation,
                'X_val': X_val_secondOrientation,
                'Y_train': Y_train_secondOrientation,
                'Y_val': Y_val_secondOrientation,
            }
