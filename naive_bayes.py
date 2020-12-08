from collections import Counter, defaultdict

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
                 "_vocabulary_N"]

    def __init__(self, database):
        """

        @:param database    object from the reader class containing the data
		"""

        # List with the words contained in positive tweets
        db_p = database.get_positive_words()

        # List with the words contained in negative tweets
        db_n = database.get_negative_words()

        # Calculation of the priori values for each target class {-,+}
        tot = len(db_n) + len(db_p)
        self._priori = (len(db_n) / tot, len(db_p) / tot)

        # Create vocabulary with {word:freq} using the Counter library
        self._vocabulary_P = dict(Counter(db_p))
        self._vocabulary_N = dict(Counter(db_n))

        # Calculus of lenghts that will be used in the likelihood calculus
        n_p = sum(self._vocabulary_P.values())
        n_n = sum(self._vocabulary_N.values())
        len_vocab = max(len(self._vocabulary_P), len(self._vocabulary_N))

        # Likelihood values for each target class
        lh_p_vals = [(nk + 1) / (n_p + len_vocab) for nk in self._vocabulary_P.values()]
        lh_n_vals = [(nk + 1) / (n_n + len_vocab) for nk in self._vocabulary_N.values()]

        # Creation of dictionaries with likelihoods with {word:likelihood}
        self._likelihood_P = dict(zip(self._vocabulary_P.keys(), lh_p_vals))
        self._likelihood_N = dict(zip(self._vocabulary_N.keys(), lh_n_vals))




    def predict(self, tweet):
        """
        Given a tweet, predicts if the tweet has a positive feeliing or negative

        :param tweet: tweet to be analized
        :return: integer indicating the feeling of the tweet. 0 for positive, 1
                 for negative
        """

        sum_priori = [0, 0]  # [sum_priori_N, sum_priori_P]

        for w in tweet:
            sum_priori[NEG] += self._likelihood_N[w]
            sum_priori[POS] += self._likelihood_P[w]

        sum_priori[NEG] += self._priori[NEG]
        sum_priori[POS] += self._priori[POS]

        return sum_priori.index(max(sum_priori))




################################################################################
############################## PUBLIC METHODS ##################################
################################################################################