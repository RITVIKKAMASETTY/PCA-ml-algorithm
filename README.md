# Principal Component Analysis (PCA) Implementation

## Overview

This repository contains an implementation of Principal Component Analysis (PCA), a dimensionality reduction technique widely used in machine learning and data science. PCA transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible.

## Table of Contents

- [Theory](#theory)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Visualization](#visualization)
- [Performance Considerations](#performance-considerations)
- [Contributing](#contributing)
- [License](#license)

## Theory

Principal Component Analysis (PCA) is a statistical procedure that uses orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.

### Key Concepts

1. **Dimensionality Reduction**: PCA reduces the number of dimensions by projecting data onto a lower-dimensional subspace.

2. **Variance Preservation**: PCA finds directions (principal components) that maximize the variance in the dataset.

3. **Eigenvalues and Eigenvectors**: The eigenvectors of the covariance matrix represent the directions of maximum variance, and the eigenvalues represent the magnitude of this variance.

### Algorithm Steps

1. **Standardization**: Center the data by subtracting the mean and optionally scale by dividing by the standard deviation.

2. **Covariance Matrix Computation**: Calculate the covariance matrix to understand how features vary with respect to each other.

3. **Eigendecomposition**: Compute the eigenvalues and eigenvectors of the covariance matrix.

4. **Feature Vector**: Select top k eigenvectors (based on eigenvalues) to form a feature vector.

5. **Transformation**: Project the original data onto the new subspace.

## Installation

```bash
# Using pip
pip install pca-implementation

# From source
git clone https://github.com/username/pca-implementation.git
cd pca-implementation
pip install -e .
```

## Usage

### Basic Example

```python
import numpy as np
from pca_implementation import PCA

# Create sample data
X = np.array([
    [2.5, 2.4], 
    [0.5, 0.7], 
    [2.2, 2.9], 
    [1.9, 2.2], 
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

# Initialize PCA with 1 component
pca = PCA(n_components=1)

# Fit and transform data
X_reduced = pca.fit_transform(X)

# Get explained variance ratio
explained_variance = pca.explained_variance_ratio_

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance ratio: {explained_variance}")

# Transform back to original space (reconstruction)
X_reconstructed = pca.inverse_transform(X_reduced)
```

### Advanced Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from pca_implementation import PCA
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Initialize PCA with automatic component selection
# Keep components that explain 95% of variance
pca = PCA(n_components=0.95, svd_solver='full')

# Fit and transform
X_reduced = pca.fit_transform(X)

print(f"Original dimensions: {X.shape}")
print(f"Reduced dimensions: {X_reduced.shape}")
print(f"Number of components selected: {pca.n_components_}")
print(f"Explained variance per component: {pca.explained_variance_ratio_}")

# Visualize the first two principal components
plt.figure(figsize=(10, 8))
for i in range(10):
    plt.scatter(
        X_reduced[y == i, 0], 
        X_reduced[y == i, 1],
        label=str(i)
    )
plt.legend()
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Digits dataset projected onto first two principal components')
plt.savefig('pca_visualization.png')
plt.show()
```

## API Reference

### `PCA` Class

```python
PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
```

#### Parameters

- `n_components` : int, float, or 'mle', default=None
  - Number of components to keep:
  - If int, the number of components to keep.
  - If float, select the number of components such that the percentage of variance retained is greater than the specified value.
  - If 'mle', use Minka's MLE to determine the number of components.
  - If None, all components are kept.

- `copy` : bool, default=True
  - If False, avoid copying data when possible.

- `whiten` : bool, default=False
  - When True, the components_ vectors are divided by the singular values and multiplied by sqrt(n_samples).

- `svd_solver` : {'auto', 'full', 'randomized'}, default='auto'
  - Method for computing the SVD:
  - 'auto': Auto-select based on data shape and n_components.
  - 'full': Use standard SVD algorithm.
  - 'randomized': Use randomized SVD algorithm.

- `tol` : float, default=0.0
  - Tolerance for singular values in SVD.

- `iterated_power` : int or 'auto', default='auto'
  - Number of iterations for the power method when using randomized SVD.

- `random_state` : int, RandomState instance or None, default=None
  - Controls randomized SVD algorithm seed.

#### Attributes

- `components_` : ndarray of shape (n_components, n_features)
  - Principal axes in feature space.

- `explained_variance_` : ndarray of shape (n_components,)
  - Amount of variance explained by each component.

- `explained_variance_ratio_` : ndarray of shape (n_components,)
  - Percentage of variance explained by each component.

- `singular_values_` : ndarray of shape (n_components,)
  - The singular values.

- `mean_` : ndarray of shape (n_features,)
  - Per-feature empirical mean.

- `n_components_` : int
  - The estimated number of components.

- `n_features_` : int
  - Number of features in the training data.

- `n_samples_` : int
  - Number of samples in the training data.

#### Methods

- `fit(X, y=None)` : Fit the model with X.
- `fit_transform(X, y=None)` : Fit the model with X and apply dimensionality reduction.
- `transform(X)` : Apply dimensionality reduction to X.
- `inverse_transform(X)` : Transform data back to original space.
- `score(X, y=None)` : Return the average log-likelihood.
- `score_samples(X)` : Return the log-likelihood of each sample.
- `get_covariance()` : Compute data covariance with the generative model.
- `get_precision()` : Compute data precision matrix with the generative model.

## Visualization

PCA results can be visualized in several ways:

1. **Scree Plot**: Plot eigenvalues to determine number of components

```python
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 
         marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.95, '95% Variance Threshold', color='red')
plt.grid(True)
plt.show()
```

2. **Biplot**: Visualize both samples and feature loadings

```python
def biplot(score, coef, labels=None):
    plt.figure(figsize=(12, 9))
    xs = score[:,0]
    ys = score[:,1]
    plt.scatter(xs, ys, c='blue', alpha=0.5)
    
    for i, (x, y) in enumerate(zip(xs, ys)):
        if labels is not None:
            plt.text(x, y, labels[i])
    
    n = coef.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    
    for i in range(n):
        plt.arrow(0, 0, coef[i,0]*scalex, coef[i,1]*scaley, 
                  color='r', alpha=0.5)
        plt.text(coef[i,0]*scalex*1.15, coef[i,1]*scaley*1.15, 
                 f"Var{i+1}", color='r')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.tight_layout()
    plt.show()

# Example usage
biplot(X_reduced[:,:2], pca.components_[:2,:], labels=y)
```

## Performance Considerations

1. **Memory Efficiency**: 
   - For large datasets, consider using incremental PCA or randomized SVD.
   - Use `copy=False` when appropriate to avoid data duplication.

2. **Computational Efficiency**:
   - For high-dimensional data with few samples, 'full' SVD is faster.
   - For many samples and few dimensions, 'randomized' SVD is faster.
   - Setting `svd_solver='auto'` will choose the most efficient algorithm.

3. **Scaling**:
   - Always standardize features before PCA when scales differ significantly.
   - Use `whiten=True` to make components have unit variance.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
