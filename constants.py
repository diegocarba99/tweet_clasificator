################################################################################
############################# EXECUTION OPTIONS ################################
################################################################################
RAW_DATASET = "database/SentimentAnalysisDataset.csv"
STEMMED_DATASET = "database/FinalStemmedSentimentAnalysisDataset.csv"
SHORT_STEMMED_DATASET = "database/FinalStemmedSentimentAnalysisDataset_Short.csv"
TRIM_DICT = False
TRIM_DATASET = False
CROSS_VAL = False
LAPLACE_SMOOTHING = False

################################################################################
############################# PROGRAM CONSTANTS ################################
################################################################################
NEG = 0  # Negative sentiment index
POS = 1  # Positive sentiment index
L = 1  # Laplace smoothing - l = 1
R = 2  # R value of Laplace smoothing - R = 2
ACCURACY = 0  # Confusion matrix index
PRECISION = 1  # Confusion matrix index
RECALL = 2  # Confusion matrix index
SPECIFICITY = 3  # Confusion matrix index
TP = 0  # Metrics matrix index
TN = 3  # Metrics matrix index
FP = 2  # Metrics matrix index
FN = 1  # Metrics matrix index
