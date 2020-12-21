import pandas as pd
from constants import *
import numpy as np


class Reader(object):
    """
    @attr   _df     Pandas dataframe containing all the data from the dataset
	"""

    __slots__ = ["_df"]

    def __init__(self, filename):
        """
        Constructor of the Reader class. Reads the data from filename and stores
        it in a pandas DataFrame

        :param filename: Name of the file containing the data
        """

        # Read the CSV file, which is separated by semicolons (;)
        self._df = pd.read_csv(filename, sep=';')

        # Shuffle the data in the DataFrame to get random distribution of data
        self._df = self._df.iloc[np.random.permutation(len(self._df))]

        # Convert all the data in the tweet column to string type
        self._df['tweetText'] = self._df['tweetText'].apply(str)


    def create_sets(self, tam_train = 0.75):
        """
        Creates the train and test sets from an initial dataset. The train set
        is returned as two list of words, each one for one class. The test set
        is returned ad a pandas DataFrame

        :param tam_train: percentage of the train set over the whole dataset
        :return: list of positive words, list of negative words and test set
        """

        # Train/test percentage calculation
        perc = int(self._df.shape[0] * tam_train)

        # Division of the whole dataset into train and test sets using holdout
        train = self._df.iloc[0:perc, :]
        test = self._df.iloc[perc:, :]

        # Convert to list of tweets for every class
        list_p = train[train['sentimentLabel'] == 1]['tweetText'].tolist()
        list_n = train[train['sentimentLabel'] == 0]['tweetText'].tolist()

        positive_words, negative_words = [], []

        # Convert tweets into list of words
        for sentence in list_p:
            positive_words.extend(sentence.split())
        for sentence in list_n:
            negative_words.extend(sentence.split())

        return negative_words, positive_words, test

    def create_sets_CV(self, k, i):
        """
        Creates the train and test sets from an initial dataset. The train set
        is returned as two list of words, each one for one class. The test set
        is returned ad a pandas DataFrame. The function divides the sets
        depending on the step of the cross-validation.

        :param k: k-fold parammeter for the cross validation
        :param i: iteraion of the cross validation phase
        :return:
        """

        # Train/test percentage calculation
        chunk = int((self._df.shape[0] / k) + (1 if self._df.shape[0] % k > i else 0))

        # Division of the whole dataset into train and test sets using cross-val
        test = self._df.iloc[i * chunk:(i + 1) * chunk]
        train = self._df.drop(range(i*chunk, (i+1)*chunk), axis=0)

        # Convert to list of tweets for every class
        list_p = train[train['sentimentLabel'] == 1]['tweetText'].tolist()
        list_n = train[train['sentimentLabel'] == 0]['tweetText'].tolist()

        positive_words, negative_words = [], []

        # Convert tweets into list of words
        for sentence in list_p:
            positive_words.extend(sentence.split())
        for sentence in list_n:
            negative_words.extend(sentence.split())

        return negative_words, positive_words, test

