import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def import_track(id):
    script_dir = os.path.dirname(__file__)  # Script directory
    full_path = os.path.join(script_dir, '../datasets/training/')
    loc = full_path + id
    music_df = pd.read_csv(loc, sep=",", header=None)
    return music_df


def import_labels():
    script_dir = os.path.dirname(__file__)  # Script directory
    full_path = os.path.join(script_dir, '../datasets/labels.csv')
    labels_df = pd.read_csv(full_path, sep=",")
    return labels_df

class gaussian:
    pass

if __name__ == "__main__":
    # Import labels
    id_labels_df = import_labels()
    ids = id_labels_df.filter(items=['id'])
    # Import of library
    id = ids.iloc[0]
    music_df = import_track(id)