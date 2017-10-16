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


class KnnClassifier_v1:
    '''Set of tools to classify 12-dimensions features pack using k-nearest neighbors and gaussian kernel'''

    def __init__(self):
        # Initialize tuples for categories' names
        self.categories = ("classical", "country", "edm_dance", "jazz", "kids",
                           "latin", "metal", "pop", "rnb", "rock")
        self.feature_set = {}

    def gauss_kernel(self, dis_genre, std_dis):
        '''Apply gaussian kernel to distance value to get its weight in the decision'''
        return np.exp(-dis_genre / (2 * (std_dis ** 2)))

    def get_mean_vectors(self, category, set):
        '''Get mean feature vector to describe a genre'''
        mean_vector = []
        for track in set[category]['features']:
            mean_vector.append(np.mean(track, axis=0))
        return np.array(mean_vector)

    def genre_from_euclidean(self, track_features, k):
        '''Get genre from k-nearest neighbors with euclidean distance and gaussian weighing kernel'''
        distance = {}
        genre_df = []
        print 'Next Song'
        for key in self.categories:
            # Take only 300 features of each song to represent whole song
            distance[key] = sp.distance.cdist(track_features[:200], self.feature_set[key], 'sqeuclidean')

        for vector in range(len(track_features[:200])):
            feature_dis_df = []
            for key in self.categories:
                # Regroup distance of genres for vector
                data = pd.DataFrame({"Genre": key, "Distance": distance[key][vector]})
                feature_dis_df.append(data)
            # Keep n smallest distances with their genres
            feature_dis_df = pd.concat(feature_dis_df)
            std = np.std(feature_dis_df['Distance'])
            feature_dis_df = feature_dis_df.nsmallest(k, 'Distance')
            # Apply weighting kernel
            feature_dis_df['Distance'] = self.gauss_kernel(feature_dis_df['Distance'], np.sqrt(std))
            # Extract genre having more weight in the neighbours
            feature_dis_df = (feature_dis_df.groupby('Genre')['Distance'].sum())
            genre_df.append(feature_dis_df.argmax())
        genre_df = (pd.DataFrame(genre_df)).mode()[0]

        return genre_df.values[0]

    def create_results_df(self, predictions):
        '''Create dataframe with classification results'''
        return pd.DataFrame(predictions, columns=['id', 'category'])

    def class_test_set(self, test_set):
        '''Classes value of test set and offers feedback on accuracy'''
        predictions = []
        results = []
        for key in test_set:
            # Get track's id and its features and pass them through classifier
            for track, id_f, category in zip(test_set[key]['features'], test_set[key]['id'],
                                             test_set[key]['category']):
                start_time = time.time()
                prediction = self.genre_from_euclidean(track, 11)
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
        '''Classes value of unknown songs in validation set'''
        predictions = []
        results = []
        # Get track's id and its features and pass them through classifier
        for track, id_f in zip(val_set['features'], val_set['id']):
            prediction = self.genre_from_euclidean(track, 11)
            predictions.append([id_f, prediction])
        print 'Genre of ' + id_f + ' is ' + prediction
        return predictions


if __name__ == "__main__":
    # Routine to extract either the accuracy of the classifier on test set or predictions of unknown songs by id
    if sys.argv[1] == 'use_test':
        USE_VALIDATION_SET = False
    elif sys.argv[1] == 'use_val':
        USE_VALIDATION_SET = True
    else:
        print 'Wrong selection, select use_test or use_val'
        sys.exit()
    TRAIN_TEST_RATIO = 1.0
    # Initialize mean vector and covariance dictionnaries
    mean_vector = {}
    covariance_matrix = {}
    # Create class to import features
    print 'IMPORTING FEATURES...'
    features = FeaturesImport_v0()
    features.USE_PCA = False
    # Create predictions accuracy list
    results = []
    # Get ids for each categories randomly splited between training and testing datasets
    if USE_VALIDATION_SET:
        ratio = 1.0
        features.TRAINING_RATIO = 1.00
    else:
        ratio = TRAIN_TEST_RATIO
        features.TRAINING_RATIO = 0.66
    training_labels, test_labels = features.select_subset_and_split(ratio)
    # Build features dictionary for training set and testing set
    training_set = features.get_feature_dict(training_labels)
    test_set = features.get_feature_dict(test_labels)
    validation_set = features.import_validation_set()

    ## INSERT CLASSIFIER WITH TRAINING AND TESTING SET AS INPUT
    ## AND PREDICTIONS (id + category) / RESULTS (1 for match else 0)
    knn = KnnClassifier_v1()
    predictions = []
    # Group all features of same genre together
    for key in knn.categories:
        knn.feature_set[key] = np.concatenate(training_set[key]['features'])

    if USE_VALIDATION_SET:
        print 'Classifying validation set...'
        predictions = knn.class_val_set(validation_set)
        # Transfer results to csv file
        result_df = knn.create_results_df(predictions)
        result_df.to_csv(features.get_path('test_labels_v1.csv'), ',', index=False)
    else:
        print 'Classifying test set...'
        predictions, results = knn.class_test_set(test_set)
        # Output accuracy of classification
        accuracy = float(sum(results) / len(results))
        print str('\nClassifier has accuracy of ' + str(accuracy * 100) + '% over testing set')
        ################################################################
