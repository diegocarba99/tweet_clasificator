#!/usr/bin/python

from reader import *
from naive_bayes import *
from constants import *

import matplotlib.pyplot as plt
import numpy as np
import gc

if __name__ == '__main__':

    n = 1000000  # Number of rows to read from the database
    tam_train = 0.70
    tam_dict = 40000

    scores = []
    train_range = np.arange(0.7, 0.9, 0.05)
    dict_range = range(1000, 300000, 59800)

    reader = Reader(STEMMED_DATASET, n)

    #for tam_train in train_range:
    #    print("-[", tam_train, "]-")
    #    scores = []
    #    for tam_dict in dict_range:
    #        print("+[", tam_dict, "]+")
    positive_words, negative_words, test_set = reader.create_sets(tam_train)
    bayes = NaiveBayes(positive_words, negative_words, tam_dict)
    # bayes.predict(test_set)
    # bayes.calculate_metrics()
    # bayes.print_confusion_matrix()
    # bayes.print_metrics()
    # scores.append(bayes.get_acc())
    # bayes.clear()
    #        del(positive_words)
    #        del(negative_words)
    #        del(test_set)
    #        del(bayes)
    #        gc.collect()
    #    plt.plot(dict_range, scores, label=f'T{tam_train}')
    #    print(scores)

    #plt.xlabel('Dictionary size')
    #plt.ylabel('Accuracy')
    #plt.title('For every train set size accuracy score')
    #plt.legend()
    #plt.show()


