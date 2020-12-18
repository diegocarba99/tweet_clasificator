#!/usr/bin/python

from reader import *
from naive_bayes import *
from constants import *

if __name__ == '__main__':

    n = 100000  # Number of rows to read from the database
    tam_train = 0.75
    tam_dict = 40000


    reader = Reader(STEMMED_DATASET, n)

   #for tam_train in range(0.5, 0.9, 0.05):
   #    for tam_dict in range(1000, 300000, 29900)
    positive_words, negative_words, test_set = reader.create_sets(tam_train)
    bayes = NaiveBayes(positive_words, negative_words, tam_dict)
    bayes.predict(test_set)
    bayes.calculate_metrics()
    bayes.print_confusion_matrix()
    bayes.print_metrics()


