import pandas as pd
import numpy as np
import glob
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class Processor:
    def __init__(self, seed, raw_data_dir, artifact_dir):
        self.seed = seed
        self.raw_data_dir = raw_data_dir
        self.artifact_dir = artifact_dir
        # placeholders for future attributes
        self.d, self.target_var = (None, None)

    def _read_and_combine_raw_data(self):
        """
        Read all files in the provided data directory to Pandas dataframes.
        Assumes all files have the same columns, which was verified in scripts/eda.py
        """
        self.d = pd.DataFrame()
        for file in glob.glob(self.raw_data_dir+"/*.csv"):
            df = pd.read_csv(file, header=None, low_memory=False)
            self.d = pd.concat([self.d, df], axis=0)
        self.d = self.d.sample(frac=1)  # shuffle data
        self.d.reset_index(drop=True, inplace=True)

    def _identify_target_for_training(self):
        """
        Gets the name of the target variable column.
        """
        self.target_var = self.d.columns[-1]

    def _separate_prod_data(self):
        """
        Separate some records to simulate production API calls with later
        """
        if self.target_var is None:
            self._identify_target_for_training()
        prod_data = self.d.iloc[-1000:]
        prod_data.drop(self.target_var, axis=1).to_csv(self.artifact_dir+"/prod_data.csv", index=False)
        self.d = self.d.iloc[:-1000]

    def _standardize_target_classes_for_training(self):
        """
        The MIT and PTB datasets have different numbers of classes for the target variable.  There are 2 options
        to standardize them:
            1. Cluster observations around MIT class labels to create new labels for PTB.  This would expand
               the classes from 2 to 5.
            2. Re-label the MIT to normal/abnormal.  This would reduce the classes from 5 to 2.
        This method takes the 2nd option.  For documentation on the MIT dataset, to understand what the class
        labels mean, see:
            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4897569/
            https://www.hindawi.com/journals/cmmm/2018/1380348/tab1/
            https://www.kaggle.com/shayanfazeli/heartbeat/discussion/62761

        :return: Combined Pandas dataframe with standardized targets
        """
        if self.target_var is None:
            self._identify_target_for_training()
        self.d[self.target_var] = np.where(self.d[self.target_var] > 0.0, 1, 0)
        self.d[self.target_var] = np.where(self.d[self.target_var] > 0.0, 1, 0)
        return self.d

    def _create_train_test_split(self, test_size=0.25):
        """
        Create new train/test split
        """
        return train_test_split(
            self.d.drop(self.target_var, axis=1), self.d[self.target_var],
            test_size=test_size, random_state=self.seed
        )

    @staticmethod
    def normalize_data(scaler_dir, X):
        """
        Scales data to range 0-1, saves scaler to pickled object if there isn't one already

        :param X: the X train or test values
        """
        if os.path.exists(scaler_dir+"/scaler.pkl"):
            with open(scaler_dir+"/scaler.pkl", "rb") as scaler_file:
                scaler = pickle.load(scaler_file)
            X = scaler.transform(X)
        else:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            with open(scaler_dir+"/scaler.pkl", "wb") as scaler_file:
                pickle.dump(scaler, scaler_file)
        return X

    @staticmethod
    def reshape_data(X, Y=None):
        """
        Reshapes data to (samples, time_steps, features) to match what TF expects.
        Y is optional because production data will need to be reshaped but will not have a
        Y, although training/test data will have a target Y

        :param X: the X train or test values
        :param Y: the Y train or test values
        """
        X = np.expand_dims(X, axis=2)  # there is only 1 feature in this dataset
        if Y is not None:
            Y = Y.values
            return X, Y
        else:
            return X

    def prepare_training_data(self):
        """
        Runs all data preparation steps in order for training runs.

        :return: X and Y train and test data
        """
        self._read_and_combine_raw_data()
        self._separate_prod_data()
        self._standardize_target_classes_for_training()
        x_train, x_test, y_train, y_test = self._create_train_test_split()
        x_train = self.normalize_data(scaler_dir=self.artifact_dir, X=x_train)
        x_test = self.normalize_data(scaler_dir=self.artifact_dir, X=x_test)
        x_train, y_train = self.reshape_data(X=x_train, Y=y_train)
        x_test, y_test = self.reshape_data(X=x_test, Y=y_test)
        del self.d  # free up memory
        return x_train, x_test, y_train, y_test
