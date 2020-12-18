import pandas as pd
from constants import *

"""

"""


class Reader:
    """
    @attr   _df                 Pandas dataframe containing all the data from the dataset
    @attr   _positive_words     List containing all the words from positive tweets
    @attr   _negative_words     List containing all the words from negative tweets
	"""

    __slots__ = ["_df", "_positive_words", "_negative_words"]

    def __init__(self, database, n):
        """
		"""

        # Read the CSV file, which is separated by semicolons (;)
        if TRIM_DATASET:
            self._df = pd.read_csv(database, sep=';', nrows=n)
        else:
            self._df = pd.read_csv(database, sep=';')

        self._df.sample(frac=1, random_state=0)
        self._df['tweetText'] = self._df['tweetText'].apply(str)



    def create_sets(self, tam_train):

        # Train/test percentage calculation
        perc = int(self._df.shape[0] * tam_train)

        # Division of the whole dataset into train and test sets using holdout
        train = self._df.iloc[0:perc, :]
        test = self._df.iloc[perc:, :]

        # Select positive
        list_p = train[train['sentimentLabel'] == 1]['tweetText'].tolist()
        list_n = train[train['sentimentLabel'] == 0]['tweetText'].tolist()

        positive_words = []
        negative_words = []

        for sentence in list_p:
            positive_words.extend(sentence.split())

        for sentence in list_n:
            negative_words.extend(sentence.split())

        #for e in train['tweetText']:
        #    print(e)

        #for e in test['tweetText']:
        #    print(e)

        print(f'shape train - {train.shape}')
        print(f'shape test - {test.shape}')


        return negative_words, positive_words, test






    def get_positive_words(self):
        return self._positive_words

    def get_negative_words(self):
        return self._negative_words



################################################################################
############################## PUBLIC METHODS ##################################
################################################################################