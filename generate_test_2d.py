"""
Generate various 2D data distributions for visualizing train/test splits.

Each generator returns an (N, 2) numpy array suitable for passing directly
to the splitter functions as embeddings.
"""

import numpy as np


def make_unimodal(n=500, center=(0, 0), std=1.0, seed=42):
    """Single Gaussian blob."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=center, scale=std, size=(n, 2))


def make_bimodal(n=500, centers=((-3, 0), (3, 0)), std=0.8, seed=42):
    """Two separated Gaussian blobs."""
    rng = np.random.default_rng(seed)
    per = n // 2
    blobs = [rng.normal(loc=c, scale=std, size=(per, 2)) for c in centers]
    # handle odd n
    if n % 2:
        blobs.append(rng.normal(loc=centers[0], scale=std, size=(1, 2)))
    return np.concatenate(blobs)


def make_multimodal(n=600, n_clusters=4, radius=4.0, std=0.6, seed=42):
    """Multiple Gaussian blobs arranged in a circle."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    centers = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    per = n // n_clusters
    blobs = [rng.normal(loc=c, scale=std, size=(per, 2)) for c in centers]
    return np.concatenate(blobs)


def make_concentric_rings(n=600, radii=(1.0, 3.0), noise=0.2, seed=42):
    """Concentric rings (annuli)."""
    rng = np.random.default_rng(seed)
    per = n // len(radii)
    parts = []
    for r in radii:
        theta = rng.uniform(0, 2 * np.pi, per)
        rr = r + rng.normal(0, noise, per)
        parts.append(np.column_stack([rr * np.cos(theta), rr * np.sin(theta)]))
    return np.concatenate(parts)


def make_moons(n=500, noise=0.1, seed=42):
    """Two interleaving half-circles."""
    rng = np.random.default_rng(seed)
    per = n // 2
    theta1 = np.linspace(0, np.pi, per)
    theta2 = np.linspace(0, np.pi, n - per)
    x1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    x2 = np.column_stack([np.cos(theta2) + 0.5, -np.sin(theta2) + 0.5])
    data = np.concatenate([x1, x2])
    data += rng.normal(0, noise, data.shape)
    return data


def make_spiral(n=500, noise=0.3, seed=42):
    """Two interleaving spirals."""
    rng = np.random.default_rng(seed)
    per = n // 2
    t = np.linspace(0, 3 * np.pi, per)
    x1 = np.column_stack([t * np.cos(t), t * np.sin(t)])
    x2 = np.column_stack([t * np.cos(t + np.pi), t * np.sin(t + np.pi)])
    data = np.concatenate([x1, x2])
    data += rng.normal(0, noise, data.shape)
    return data


def make_anisotropic(n=500, seed=42):
    """Elongated / skewed Gaussian (anisotropic covariance)."""
    rng = np.random.default_rng(seed)
    cov = [[3.0, 1.5], [1.5, 1.0]]
    return rng.multivariate_normal([0, 0], cov, size=n)


def make_uniform_square(n=500, low=-5, high=5, seed=42):
    """Uniform distribution over a square region."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, 2))


def make_density_gradient(n=500, seed=42):
    """Non-uniform density: dense on left, sparse on right."""
    rng = np.random.default_rng(seed)
    x = rng.exponential(scale=2.0, size=n)
    y = rng.normal(0, 1, size=n)
    return np.column_stack([x, y])


def make_clustered_outliers(n=500, n_outliers=30, seed=42):
    """Main cluster with distant outlier points."""
    rng = np.random.default_rng(seed)
    main = rng.normal(0, 1, size=(n - n_outliers, 2))
    outliers = rng.uniform(-10, 10, size=(n_outliers, 2))
    return np.concatenate([main, outliers])


ALL_GENERATORS = {
    "unimodal": make_unimodal,
    "bimodal": make_bimodal,
    "multimodal": make_multimodal,
    "concentric_rings": make_concentric_rings,
    "moons": make_moons,
    "spiral": make_spiral,
    "anisotropic": make_anisotropic,
    "uniform_square": make_uniform_square,
    "density_gradient": make_density_gradient,
    "clustered_outliers": make_clustered_outliers,
}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, (name, gen) in zip(axes.flat, ALL_GENERATORS.items()):
        data = gen()
        ax.scatter(data[:, 0], data[:, 1], s=5, alpha=0.6)
        ax.set_title(name)
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("test_data/2d_distributions.png", dpi=150)
    plt.show()
    print("Saved preview to test_data/2d_distributions.png")
