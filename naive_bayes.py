from collections import Counter, defaultdict
from constants import *
import math


class NaiveBayes:
    """
	@attr   _priori             List/tuple containing the priori values P(Vj)
	                            for all the target classes in the Bayesian
	                            network
	@attr   _likelihood_P       Dictionary containing the likelihood values for
	                            the target class 'Positive' - P(Wj|+). The key
	                            is the word and the value is the frequency
	                            {word:freq}
	@attr   _likelihood_N       Dictionary containing the likelihood values for
	                            the target class 'Negative' - P(Wj|-). The key
	                            is the word and the value is the frequency
	                            {word:freq}
	@attr	_vocabulary_P       Dictionary containing all the words in the
	                            database with their frequency corresponding to
	                            the 'Positive' target class. The key is the word
	                            and the value is the frequency {word:freq}
	@attr	_vocabulary_N       Dictionary containing all the words in the
	                            database with their frequency corresponding to
	                            the 'Positive' target class. The key is the word
	                            and the value is the frequency {word:freq}
    @attr   _conf_matrix        Confusion matrix of the models results
    @attr   _metrics            Metrics matrix of the models results
    @attr   _len_vocab          Length of the whole word vocabulary
	"""

    __slots__ = ["_priori", "_lhP", "_lhN", "_vocabP","_vocabN",
                 "_conf_matrix", "_metrics", "_len_vocab"]

    def __init__(self, ln, lp, tam_dict=1.0):
        """
        Constructor of the NaiveBayes class. Creates a vocabulary and likelihood
        dictionary for the positive and negative classes.

        :param ln: list of the words in negative tweets
        :param lp: list of the words in positive tweets
        :param tam_dict: percentage of the dictionary to trim up to
        """

        # Calculation of the priori values for each target class {-,+}
        tot = len(ln) + len(lp)
        self._priori = (len(ln) / tot, len(lp) / tot)

        # Create vocabulary with {word:freq} using the Counter library
        self._vocabP = dict(Counter(lp))
        self._vocabN = dict(Counter(ln))

        # Order dictionary by value
        self._vocabP = dict(sorted(self._vocabP.items(), key=lambda item: item[1], reverse=True))
        self._vocabN = dict(sorted(self._vocabN.items(), key=lambda item: item[1], reverse=True))

        # Calculus of lengths that will be used in the likelihood calculus
        n_p = sum(self._vocabP.values())
        n_n = sum(self._vocabN.values())
        self._len_vocab = max(len(self._vocabP), len(self._vocabN))

        # Likelihood values for each target class
        lh_p_vals = [nk / (n_p + self._len_vocab) for nk in self._vocabP.values()]
        lh_n_vals = [nk / (n_n + self._len_vocab) for nk in self._vocabN.values()]

        # Creation of dictionaries with likelihoods with {word:likelihood}
        self._lhP = dict(zip(self._vocabP.keys(), lh_p_vals))
        self._lhN = dict(zip(self._vocabN.keys(), lh_n_vals))

        # Size of the trimming of the vocabulary
        tam_p = int(len(self._lhP) * tam_dict)
        tam_n = int(len(self._lhN) * tam_dict)

        # Trimming the dictionary up to a length
        if TRIM_DICT:
            self._lhP = dict(list(self._lhP.items())[:tam_p])
            self._lhN = dict(list(self._lhN.items())[:tam_n])

        # Initialization of the matrices
        self._conf_matrix = [0, 0, 0, 0]
        self._metrics = [0, 0, 0, 0]

    def classify(self, tweet):
        """
        Given a tweet, predicts if the tweet has a positive feeling or negative

        :param tweet: tweet to be analyzed
        :return: integer indicating the feeling of the tweet. 0 for positive, 1
                 for negative
        """

        sum_priori = [0, 0]  # [sum_priori_N, sum_priori_P]

        for w in tweet.split():
            if w in self._lhN:
                sum_priori[NEG] += math.log(self._lhN[w])
            elif LAPLACE_SMOOTHING:
                sum_priori[NEG] += math.log(1/self._len_vocab)

            if w in self._lhP:
                sum_priori[POS] += math.log(self._lhP[w])
            elif LAPLACE_SMOOTHING:
                sum_priori[POS] += math.log(1 / self._len_vocab)

        sum_priori[NEG] += math.log(self._priori[NEG])
        sum_priori[POS] += math.log(self._priori[POS])

        return sum_priori.index(max(sum_priori))

    def predict(self, test):
        """
        Given a test set, predicts which class do each of the tweets correspond
        to and calculates the confusion matrix

        :param test: Pandas DataFrame containing the test set
        """

        # For every tweet in the test set
        for i in range(test.shape[0]):

            # Classify tweet
            tweet = self.classify(test.iloc[i, 1])

            # Ground truth class
            clase = test.iloc[i, 3]

            # Comparison of prediction and ground truth
            if clase and tweet:
                self._conf_matrix[TP] += 1
            elif clase and not tweet:
                self._conf_matrix[FN] += 1
            elif not clase and tweet:
                self._conf_matrix[FP] += 1
            elif not clase and not tweet:
                self._conf_matrix[TN] += 1

    def calculate_metrics(self):
        """
        After predicting a test set, calculate the performance metrics
        """

        # Get the confusion matrix values
        tp = self._conf_matrix[TP]
        fn = self._conf_matrix[FN]
        fp = self._conf_matrix[FP]
        tn = self._conf_matrix[TN]

        # Calculate performance metrics
        try:
            self._metrics[ACCURACY] = (tp + tn) / (tp + tn + fp + fn)
        except ZeroDivisionError as error:
            self._metrics[ACCURACY] = 0
        try:
            self._metrics[PRECISION] = tp / (tp + fp)
        except ZeroDivisionError as error:
            self._metrics[PRECISION] = 0
        try:
            self._metrics[RECALL] = tp / (tp + fn)
        except ZeroDivisionError as error:
            self._metrics[RECALL] = 0
        try:
            self._metrics[SPECIFICITY] = tn / (tn + fp)
        except ZeroDivisionError as error:
            self._metrics[SPECIFICITY] = 0



    def print_confusion_matrix(self):
        """
        Print the confusion matrix in a pretty way
        """
        print("--- CONFUSION MATRIX ------------------------------------------")
        print(f'TP:{self._conf_matrix[TP]} | FN:{self._conf_matrix[FN]}')
        print(f'FP:{self._conf_matrix[FP]} | TN:{self._conf_matrix[TN]}')


    def print_metrics(self):
        """
        Print the performance metrics in a pretty way
        """
        print("--- METRICS ---------------------------------------------------")
        print(f'ACC{self._metrics[ACCURACY]} \t PRE:{self._metrics[PRECISION]}')
        print(f'REC:{self._metrics[RECALL]} \t SPE:{self._metrics[SPECIFICITY]}')

    def print_simple_metrics(self):
        """
        Print the confusion matrix in a simple way
        """
        print(f'ACC[{self._metrics[ACCURACY]:.2f}] \t PRE[{self._metrics[PRECISION]:.2f}] \t REC[{self._metrics[RECALL]:.2f}] \t SPE[{self._metrics[SPECIFICITY]:.2f}]')

    def print_simple_matrix(self):
        """
        Print the performance metrics in a pretty way
        """
        print(f'TP[{self._conf_matrix[TP]}] - FN[{self._conf_matrix[FN]}] - FP[{self._conf_matrix[FP]}] - TN[{self._conf_matrix[TN]}]')


    def get_acc(self):
        """
        Get the accuracy of the tested method
        :return: accuracy of the method
        """
        return self._metrics[ACCURACY]




