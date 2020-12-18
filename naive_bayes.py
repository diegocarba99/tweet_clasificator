from collections import Counter, defaultdict
from constants import *
import math

import json

"""

"""


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
	"""

    __slots__ = ["_priori", "_likelihood_P","_likelihood_N", "_vocabulary_P",
                 "_vocabulary_N", "_confusion_matrix", "_metrics", "_len_vocab"]

    def __init__(self, db_n, db_p, tam_dict):
        """

        @:param database    object from the reader class containing the data
		"""

        # Calculation of the priori values for each target class {-,+}
        tot = len(db_n) + len(db_p)
        self._priori = ( (len(db_n)+L) / ((tot)+L*R), (len(db_p)+L) / ((tot)+L*R))

        # Create vocabulary with {word:freq} using the Counter library
        self._vocabulary_P = dict(Counter(db_p))
        self._vocabulary_N = dict(Counter(db_n))

        # Order dictionary by value
        self._vocabulary_P = dict(sorted(self._vocabulary_P.items(), key=lambda item: item[1]))
        self._vocabulary_N = dict(sorted(self._vocabulary_N.items(), key=lambda item: item[1]))

        # Triming the dictionary up to a length
        self._vocabulary_P = dict(list(self._vocabulary_P .items())[:tam_dict])
        self._vocabulary_N = dict(list(self._vocabulary_N.items())[:tam_dict])


        # Calculus of lenghts that will be used in the likelihood calculus
        n_p = sum(self._vocabulary_P.values())
        n_n = sum(self._vocabulary_N.values())
        len_vocab = max(len(self._vocabulary_P), len(self._vocabulary_N))
        self._len_vocab = len_vocab

        # Likelihood values for each target class
        lh_p_vals = [(nk + L) / (n_p + len_vocab) for nk in self._vocabulary_P.values()]
        lh_n_vals = [(nk + L) / (n_n + len_vocab) for nk in self._vocabulary_N.values()]

        # Creation of dictionaries with likelihoods with {word:likelihood}
        self._likelihood_P = dict(zip(self._vocabulary_P.keys(), lh_p_vals))
        self._likelihood_N = dict(zip(self._vocabulary_N.keys(), lh_n_vals))

        print("likelihood negativo length - ", len(self._likelihood_N))
        print("likelihood positivo length - ", len(self._likelihood_P))




        self._confusion_matrix = [0, 0, 0, 0]
        self._metrics = [0, 0, 0, 0]

    ############################################################################
    ############################## PUBLIC METHODS ##############################
    ############################################################################

    def classify(self, tweet):
        """
        Given a tweet, predicts if the tweet has a positive feeling or negative

        :param tweet: tweet to be analyzed
        :return: integer indicating the feeling of the tweet. 0 for positive, 1
                 for negative
        """

        sum_priori = [0, 0]  # [sum_priori_N, sum_priori_P]

        pp, pn = [],[]

        for w in tweet.split():
            if w in self._likelihood_N:
                sum_priori[NEG] += math.log(self._likelihood_N[w])
                pn.append(math.log(self._likelihood_N[w]))
            else:
                sum_priori[NEG] += math.log(1/self._len_vocab)
                pn.append(math.log(1/self._len_vocab))
            if w in self._likelihood_P:
                sum_priori[POS] += math.log(self._likelihood_P[w])
                pp.append(math.log(self._likelihood_P[w]))
            else:
                sum_priori[POS] += math.log(1 / self._len_vocab)
                pp.append(math.log(1 / self._len_vocab))

        sum_priori[NEG] += math.log(self._priori[NEG])
        pn.append(math.log(self._priori[NEG]))

        sum_priori[POS] += math.log(self._priori[POS])
        pp.append(math.log(self._priori[POS]))


       # #for e in pn:
        #    print(e, end=' + ')
#        print("{0:.3f}".format(sum_priori[NEG]))
 #       for e in pp:
  #          print(e, end=' + ')
   #     print("{0:.3f}".format(sum_priori[POS]), end='\n')

    #    print(1/self._len_vocab, math.log(1/self._len_vocab))



        return sum_priori.index(max(sum_priori))


    def predict(self, test):

        #print("Shape of test", test.shape)
        for i in range(test.shape[0]):
            tweet = self.classify(test.iloc[i, 1])
            clase = test.iloc[i,3]

            #print(clase,"-",tweet, end='\n\n')

            if clase and tweet:
                self._confusion_matrix[TP] += 1
            elif clase and not tweet:
                self._confusion_matrix[FN] += 1
            elif not clase and tweet:
                self._confusion_matrix[FP] += 1
            elif not clase and not tweet:
                self._confusion_matrix[TN] += 1


    def calculate_metrics(self):

        tp = self._confusion_matrix[TP]
        fn = self._confusion_matrix[FN]
        fp = self._confusion_matrix[FP]
        tn = self._confusion_matrix[TN]

        metrics = [None] * 4

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

        return self._metrics


    def print_confusion_matrix(self):
        print("--- CONFUSION MATRIX ------------------------------------------")
        print(f'TP:{self._confusion_matrix[TP]} | FN:{self._confusion_matrix[FN]}')
        print(f'FP:{self._confusion_matrix[FP]} | TN:{self._confusion_matrix[TN]}')


    def print_metrics(self):
        print("--- METRICS ---------------------------------------------------")
        print(f'Accuracy:    {self._metrics[ACCURACY]}')
        print(f'Precision:   {self._metrics[PRECISION]}')
        print(f'Recall:      {self._metrics[RECALL]}')
        print(f'Specificity: {self._metrics[SPECIFICITY]}')

    ############################################################################
    ############################## PRIVATE METHODS #############################
    ############################################################################7


    def print_dicts(self):
        print(json.dumps(self._likelihood_P, indent=1))

        print(json.dumps(self._likelihood_N, indent=1))


