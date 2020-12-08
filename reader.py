import pandas as pd

"""

"""


class Reader:
    """
    @attr   _df                 Pandas dataframe containing all the data from the dataset
    @attr   _positive_words     List containing all the words from positive tweets
    @attr   _negative_words     List containing all the words from negative tweets
	"""

    __slots__ = ["_df", "_positive_words", "_negative_words"]

    def __init__(self, database):
        """
		"""

        # Read the CSV file, which is separated by semicolons (;)
        self._df = pd.read_csv(database, sep=';')

        # Select positive
        list_p = self._df.loc[self._df['sentimentLabel'] == 1]['tweetText'].tolist()
        list_n = self._df.loc[self._df['sentimentLabel'] == 0]['tweetText'].tolist()

        self._positive_words = []
        self._negative_words = []

        for sentence in list_p:
            if isinstance(sentence, str):
                self._positive_words.extend(sentence.split())

        for sentence in list_n:
            if isinstance(sentence, str):
                self._negative_words.extend(sentence.split())

    def get_positive_words(self):
        return self._positive_words

    def get_negative_words(self):
        return self._negative_words



################################################################################
############################## PUBLIC METHODS ##################################
################################################################################