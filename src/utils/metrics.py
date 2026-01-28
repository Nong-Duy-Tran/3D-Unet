import numpy as np


def get_class_weights(labels):
    """Calculate class weights for imbalanced datasets."""
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    total_samples = len(labels)

    class_weights = total_samples / (len(class_counts) * class_counts)

    return class_weights
