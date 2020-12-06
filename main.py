#!/usr/bin/python

from .reader import *
from .naive_bayes import *
from .constants import *

if __name__ == '__main__':

    reader = Reader(SHORT_STEMMED_DATASET)
    bayes = NaiveBayes(reader)
