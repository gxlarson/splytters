import numpy as np


def reduce_umap(data, n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=None):
    """Reduce high-dimensional data using UMAP."""
    from umap import UMAP

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(np.asarray(data))


def reduce_tsne(data, n_components=2, perplexity=30.0, learning_rate="auto", metric="euclidean", random_state=None):
    """Reduce high-dimensional data using t-SNE."""
    from sklearn.manifold import TSNE

    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(np.asarray(data))


def reduce(data, method="umap", n_components=2, **kwargs):
    """Reduce high-dimensional data using the specified method.

    Args:
        data: Array-like of shape (n_samples, n_features).
        method: "umap" or "tsne".
        n_components: Target dimensionality.
        **kwargs: Passed to the underlying reducer.

    Returns:
        np.ndarray of shape (n_samples, n_components).
    """
    if method == "umap":
        return reduce_umap(data, n_components=n_components, **kwargs)
    elif method == "tsne":
        return reduce_tsne(data, n_components=n_components, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'umap' or 'tsne'.")
