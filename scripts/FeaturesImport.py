#!/user/bin/env python
import os
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab


def import_labels():
    script_dir = os.path.dirname(__file__)  # Script directory
    full_path = os.path.join(script_dir, '../datasets/labels.csv')
    labels_df = pd.read_csv(full_path, sep=",")
    return labels_df


def select_ids(labels_df):
    return labels_df['id']


def select_categories(labels_df):
    return labels_df['category']


def import_track(relative_path,track_id):
    script_dir = os.path.dirname(__file__)  # Script directory
    full_path = os.path.join(script_dir, '../'+relative_path)
    loc = full_path + track_id
    tracks_df = pd.read_csv(loc, sep=",", header=None)
    return tracks_df


class FeaturesImport_v0():
    def __init__(self):
        # Import labels
        self.id_labels_df = import_labels()
        # Initialize tuples for categories' names
        self.categories = ("classical", "country", "edm_dance", "jazz", "kids",
                           "latin", "metal", "pop", "rnb", "rock")
        # Set training and testing ratio
        self.TRAINING_RATIO = 0.75
        # USE PCA for dimensionality reduction
        self.USE_PCA = False

    # Principal Component Analysis to reduce dimensionality of feature space
    def pca(self, feature, var_lim):
        i = 0
        max_var = 0
        results = mlab.PCA(feature)
        while max_var <= var_lim:
            max_var += results.fracs[i]
            i += 1
        return results.Y[:, 0:5]

    # Select a smaller part of the dataset's ids and split between training and testing
    def select_subset_and_split(self, ratio):
        size = self.id_labels_df.values.shape[0]
        subset_labels_df = self.id_labels_df.sample(round(size * ratio))
        training_ratio = ratio * self.TRAINING_RATIO
        training_labels_df = subset_labels_df.sample(round(size * training_ratio))
        test_labels_df = subset_labels_df.loc[~subset_labels_df['id'].isin(training_labels_df['id'])]
        return training_labels_df, test_labels_df

    def create_feature_for_category(self, category, labels_df):
        track_features = {}
        labels = labels_df.loc[labels_df['category'] == category]
        for _, label in labels.iterrows():
            features = np.array(import_track('datasets/training/', select_ids(label)))
            if self.USE_PCA:
                features = self.pca(features, 0.80)
            track_features.setdefault('features', []).append(features)
            track_features.setdefault('id', []).append(label['id'])
            track_features.setdefault('category', []).append(label['category'])
        return track_features

    def get_feature_dict(self, used_labels):
        # Initialize empty dictionnary to contain features for all labels of all categories
        features = {}
        # Fill dictionnary key with label features of related categories
        for category in self.categories:
            features[category] = self.create_feature_for_category(category, used_labels)
        return features

    def import_validation_set(self):
        track_features = {}
        fullpath = self.get_path('datasets/test/')
        ids = [f for f in os.listdir(fullpath) if os.path.isfile(os.path.join(fullpath, f))]
        for id in ids:
            loc = fullpath + id
            track_features.setdefault('features', []).append(np.array(import_track('datasets/test/', id)))
            track_features.setdefault('id', []).append(id)
        return track_features

    def get_path(self, relative_path):
        script_dir = os.path.dirname(__file__)  # Script directory
        fullpath = os.path.join(script_dir, '../' + relative_path)
        return fullpath



if __name__ == "__main__":
    # Import labels
    id_labels_df = import_labels()
    # Import of library
    music_df = import_track(select_ids(id_labels_df)[0])