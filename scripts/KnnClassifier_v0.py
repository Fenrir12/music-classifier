#!/user/bin/env python
from FeaturesImport import FeaturesImport_v0
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp
import time
import operator
import xarray as xr


class KnnClassifier_v0:
    def __init__(self):
        # Initialize tuples for categories' names
        self.categories = ("classical", "country", "edm_dance", "jazz", "kids",
                           "latin", "metal", "pop", "rnb", "rock")
        self.feature_set = {}

    # Get k-nearest euclidean distance genre of a song with training dataset
    def genre_from_euclidean(self, track_features, k):
        '''Compute distance matrix of queried song with dataset features
        and extract closest neighbors'''
        distance ={}
        genre_df = []
        print 'Next Song'
        for key in self.categories:
            distance[key] = sp.distance.cdist(track_features[:300], self.feature_set[key], 'sqeuclidean')
            distance[key] = np.take(np.partition(distance[key], k), range(k), 1)

        for vector in range(len(track_features[:300])):
            feature_dis_df = []
            for key in self.categories:
                data = pd.DataFrame({"Genre": key, "Distance": distance[key][vector]})
                feature_dis_df.append(data)
            feature_dis_df = pd.concat(feature_dis_df)
            feature_dis_df = feature_dis_df.nsmallest(k, 'Distance')
            genre_df.append(feature_dis_df['Genre'].mode()[0])
        genre_df = (pd.DataFrame(genre_df)).mode()[0]

        return genre_df.values[0]

    # Create dataframe with classification results
    def create_results_df(self, predictions):
        '''Creates dataframe with prediction results'''
        return pd.DataFrame(predictions, columns=['id', 'category'])

    # Make predictions of genre based on input test set
    def class_test_set(self, test_set):
        '''Classes value of test set and offers feedback on accuracy'''
        predictions = []
        results = []
        for key in test_set:
            # Get track's id and its features and pass them through classifier
            for track, id_f, category in zip(test_set[key]['features'], test_set[key]['id'],
                                                test_set[key]['category']):
                start_time = time.time()
                prediction = self.genre_from_euclidean(track, 55)
                predictions.append([id_f, prediction])
                if prediction == category:
                    results.append(1.0)
                    print 'Genre of ' + id_f + ' is ' + prediction \
                          + ' in ' + str(time.time() - start_time) + ' seconds' + ': right'
                else:
                    results.append(0.0)
                    print 'Genre of ' + id_f + ' is ' + prediction \
                          + ' in ' + str(time.time() - start_time) + ' seconds' + ': wrong'
        return predictions, results

    def class_val_set(self, val_set):
        '''Classes value of validation set'''
        predictions = []
        results = []
        # Get track's id and its features and pass them through classifier
        for track, id_f in zip(val_set['features'], val_set['id']):
            prediction = self.genre_from_euclidean(track, 3)
            predictions.append([id_f, prediction])

        return predictions
if __name__ == "__main__":
    val = sys.argv[1]
    USE_VALIDATION_SET = val
    TRAIN_TEST_RATIO = 0.5
    # Initialize mean vector and covariance dictionnaries
    mean_vector = {}
    covariance_matrix = {}
    # Create class to import features
    features = FeaturesImport_v0()
    # Create predictions accuracy list
    results = []
    # Get ids for each categories randomly splited between training and testing datasets
    if USE_VALIDATION_SET:
        ratio = 1.0
        features.TRAINING_RATIO = 1.0
    else:
        ratio = TRAIN_TEST_RATIO
        features.TRAINING_RATIO = 0.80
    training_labels, test_labels = features.select_subset_and_split(ratio)
    # Build features dictionary for training set and testing set
    training_set = features.get_feature_dict(training_labels)
    test_set = features.get_feature_dict(test_labels)
    validation_set = features.import_validation_set()

    ## INSERT CLASSIFIER WITH TRAINING AND TESTING SET AS INPUT
    ## AND PREDICTIONS (id + category) / RESULTS (1 for match else 0)
    knn = KnnClassifier_v0()
    predictions = []
    # Group all features of same genre together
    for key in knn.categories:
        knn.feature_set[key] = np.concatenate(training_set[key]['features'])

    if USE_VALIDATION_SET:
        predictions = knn.class_val_set(validation_set)
        # Transfer results to csv file
        result_df = knn.create_results_df(predictions)
        result_df.to_csv(features.get_path('test_labels.csv'), ',', index=False)
    else:
        predictions, results = knn.class_test_set(test_set)
        # Output accuracy of classification
        accuracy = float(sum(results) / len(results))
        print str('\nClassifier has accuracy of ' + str(accuracy * 100) + '% over testing set')
    ################################################################
