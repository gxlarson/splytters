from sklearn.neighbors import NearestNeighbors
from scipy.stats import wasserstein_distance
import numpy as np
def split_wasserstein_data(embeddings, test_size, no_trials, leaf_size):

    '''
    Finds appropriate test sets from encoded texts a.k.a embeddings through the wasserstein metric
    as described here: https://aclanthology.org/2021.eacl-main.156v2.pdf.

    Steps are as follows
        1. Embeddings are projects in a nearest neighbor structure
        2. A random centroid is sampled, from which nearest neighbors using the
           wasserstein metric is sampled.
        3. This is the new test set.
        4. Repeated for no_trials


    Parameters:
        embeddings (np.array): Embeddings of texts - One example, would be one hot encoding
        test_size (int): The size of the test set
        no_trials (int): The no trials to extrapolate the test set
        leaf_size (int): Leaf size for nearest neighbors. High values mean slow, less memory intensive computation.
                   Low values mean otherwise.

    Returns:
        indices_trials (list): List of test set indices for each trial.
    '''


    tree = NearestNeighbors(
        n_neighbors=test_size,
        algorithm="ball_tree",
        leaf_size = leaf_size,
        metric= wasserstein_distance,
    )
    tree.fit(embeddings)

    indices_trials = []

    for trial in range(no_trials):
        
        sampled_point = np.random.randint(
        embeddings.max().max() + 1, size=(1, embeddings.shape[1]))

        nearest_neighbors = tree.kneighbors(sampled_point, return_distance=False)
        nearest_neighbors = nearest_neighbors[0]
        indices_trials.append(nearest_neighbors)

    return indices_trials