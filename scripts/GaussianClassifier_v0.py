#!/user/bin/env python
from FeaturesImport import FeaturesImport_v0
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class GaussianClassifier_v0:
    def __init__(self):
        # Initialize tuples for categories' names
        self.categories = ("classical", "country", "edm_dance", "jazz", "kids",
                           "latin", "metal", "pop", "rnb", "rock")
        self.mean_vector = {}
        self.covariance_matrices = {}

    def get_mean_vector(self, category, set):
        mean_vector = []
        for track in set[category]['features']:
            mean_vector.append(np.mean(track, axis=0))
        return np.array(np.mean(mean_vector, axis=0))

    # Extract and concatenates all tracks along 12 variables
    def get_category_scatter(self, category, set):
        return np.concatenate(tuple(set[category]['features']))

    # Extract covariance matrix for each category of complete feature scatter of the genre
    def get_covariance_matrix(self, category, set):
        scatter = self.get_category_scatter(category, set)
        return np.matrix(np.cov(scatter.T))

    # Fills mean vector and covariance matrices for each genre of the classifier
    def train_gaussian(self, training_set):
        # Get mean vector and covariance matrix of data
        for category in gauss.categories:
            self.mean_vector[category] = gauss.get_mean_vector(category, training_set)
            self.covariance_matrices[category] = gauss.get_covariance_matrix(category, training_set)

    # Generator to get values of unll for feature
    def unll_gen(self, track, inv_cov, mean_vector):
        for feature in track:
            yield np.array(np.dot(np.dot(feature - mean_vector, inv_cov), (feature - mean_vector).T))

    # Returns dictionnary key of lowest mean unll value across all genres
    def eval_unll_of_track(self, track):
        # Dictionnary for unll values between feature and each genre
        unll = {}
        unll_mean = {}
        for category in gauss.categories:
            # Get inverse covariance matrix of this genre
            inv_cov = np.linalg.inv(self.covariance_matrices[category])
            mean_vector = self.mean_vector[category]
            # Calculate unll for each feature of the given track
            for new_unll in self.unll_gen(track, inv_cov, mean_vector):
                unll.setdefault(category, []).append(new_unll)
            unll_mean[category] = np.mean(unll[category])
        return min(unll_mean, key=unll_mean.get)

    # Make predictions of genre based on input test set
    def class_test_set(self, test_set):
        predictions = []
        results = []
        for key in test_set:
            # Get track's id and its features and pass them through classifier
            for features, id_f, category in zip(test_set[key]['features'], test_set[key]['id'],
                                                test_set[key]['category']):
                prediction = self.eval_unll_of_track(features)
                predictions.append([id_f, prediction])
                if prediction == category:
                    results.append(1.0)
                else:
                    results.append(0.0)
        return predictions, results

    def class_validation_set(self, val_set):
        predictions = []
        # Get track's id and its features and pass them through classifier
        for features, id_f in zip(val_set['features'], val_set['id']):
            prediction = self.eval_unll_of_track(features)
            predictions.append([id_f, prediction])
        return predictions

    # Create dataframe with classification results
    def create_results_df(self, predictions):
        return pd.DataFrame(predictions, columns=['id', 'category'])


if __name__ == "__main__":
    USE_VALIDATION_SET = True
    TRAIN_TEST_RATIO = 0.75
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
        features.TRAINING_RATIO = 0.66
    training_labels, test_labels = features.select_subset_and_split(ratio)
    # Build features dictionary for training set and testing set
    training_set = features.get_feature_dict(training_labels)
    test_set = features.get_feature_dict(test_labels)
    validation_set = features.import_validation_set()

    ## INSERT CLASSIFIER WITH TRAINING AND TESTING SET AS INPUT
    ## AND PREDICTIONS (id + category) / RESULTS (1 for match else 0)
    # Initialize gaussian classifier
    gauss = GaussianClassifier_v0()
    gauss.train_gaussian(training_set)
    # Get classification results
    if USE_VALIDATION_SET:
        predictions = gauss.class_validation_set(validation_set)
    else:
        predictions, results = gauss.class_test_set(test_set)
    ################################################################

    # Transfer results to csv file
    result_df = gauss.create_results_df(predictions)
    result_df.to_csv(features.get_path('test_labels.csv'), ',', index=False)
    # Output accuracy of classification
    accuracy = float(sum(results) / len(results))
    print str('\nClassifier has accuracy of ' + str(accuracy * 100) + '% over testing set')
