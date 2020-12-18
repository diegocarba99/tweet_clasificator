RAW_DATASET = "database/SentimentAnalysisDataset.csv"
STEMMED_DATASET = "database/FinalStemmedSentimentAnalysisDataset.csv"
SHORT_STEMMED_DATASET = "database/FinalStemmedSentimentAnalysisDataset_Short.csv"

NEG = 0
POS = 1

L = 1  # Laplace smoothing - l = 1
R = 2  # R value of Laplace smoothing - R = 2


TRIM_DICT = False
TRIM_DATASET = False


################################################################################
############################## METRICS INDEXES #################################
################################################################################
ACCURACY = 0
PRECISION = 1
RECALL = 2
SPECIFICITY = 3
TP = 0
TN = 3
FP = 2
FN = 1