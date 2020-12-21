#!/usr/bin/python

from reader import *
from naive_bayes import *
from constants import *
import time
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    # Default values
    tam_dict = 1
    tam_train = 0.75
    k = 10

    # Time performance
    start = time.time()

    reader = Reader(STEMMED_DATASET)

    if CROSS_VAL:
        scores = []
        for i in range(k):
            p_words, n_words, test_set = reader.create_sets_CV(k, i)
            bayes = NaiveBayes(p_words, n_words, tam_dict)
            bayes.predict(test_set)
            bayes.calculate_metrics()
            scores.append(bayes.get_acc())
            bayes.print_simple_matrix()
            bayes.print_simple_metrics()
        print(f'\nACC-CV: {sum(scores)/k}')

    else:
        p_words, n_words, test_set = reader.create_sets(tam_train)
        bayes = NaiveBayes(p_words, n_words, tam_dict)
        bayes.predict(test_set)
        bayes.calculate_metrics()
        bayes.print_simple_matrix()
        bayes.print_simple_metrics()

    print("--- Total time: %s seconds ---" % (time.time() - start))

