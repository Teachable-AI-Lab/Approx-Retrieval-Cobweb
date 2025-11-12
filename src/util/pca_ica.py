##---------------------------------------------------------------------
# File: pca_ica.py
# Author: Karthik Singaravadivelan, Anant Gupta
# Description: PCA + ICA whitening model for embedding normalization.
##----------------------------------------------------------------------
import numpy as np
import pickle
from sklearn.decomposition import PCA, FastICA

class PCAICAWhiteningModel:
    def __init__(self, mean: np.ndarray, pca_components: np.ndarray, ica_unmixing: np.ndarray,
                 pca_explained_var: np.ndarray, eps: float = 1e-8):
        self.mean = mean
        self.pca_components = pca_components
        self.pca_explained_var = pca_explained_var
        self.ica_unmixing = ica_unmixing
        self.eps = eps

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  mean.shape={self.mean.shape},\n"
            f"  pca_components.shape={self.pca_components.shape},\n"
            f"  pca_explained_var.shape={self.pca_explained_var.shape},\n"
            f"  ica_unmixing.shape={self.ica_unmixing.shape},\n"
            f"  eps={self.eps}\n"
            f")"
        )

    def transform(self, x: np.ndarray, is_ica = True) -> np.ndarray:
        """
        Apply PCA + ICA whitening to a single embedding or a batch.
        """
        is_single = (x.ndim == 1)
        if is_single:
            x = x[np.newaxis, :]

        # Step 1: Center
        x_centered = x - self.mean

        # Step 2: PCA projection
        x_pca = np.dot(x_centered, self.pca_components.T)
        x_pca /= np.sqrt(self.pca_explained_var + self.eps)

        # Step 3: ICA transform
        if is_ica:
            x_ica = np.dot(x_pca, self.ica_unmixing.T)
            return x_ica[0] if is_single else x_ica
        else:
            # If not using ICA, return PCA normalized output
            return x_pca[0] if is_single else x_pca

    @classmethod
    def fit(cls, X: np.ndarray, pca_dim: int = 256, eps: float = 1e-8,
            ica_max_iter: int = 5000, ica_tol: float = 1e-3):
        """
        Fit PCA → ICA whitening on embedding matrix X.
        """
        mean = X.mean(axis=0)
        X_centered = X - mean

        # Step 1: PCA
        pca = PCA(n_components=pca_dim)
        X_pca = pca.fit_transform(X_centered)
        components = pca.components_
        explained_var = pca.explained_variance_

        # Step 2: Normalize PCA output
        X_pca_normalized = X_pca / np.sqrt(explained_var + eps)

        # Step 3: ICA
        ica = FastICA(n_components=components.shape[0], whiten='unit-variance',
                      max_iter=ica_max_iter, tol=ica_tol)
        X_ica = ica.fit_transform(X_pca_normalized)

        return cls(mean, components, ica.components_, explained_var, eps)

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'mean': self.mean,
                'pca_components': self.pca_components,
                'pca_explained_var': self.pca_explained_var,
                'ica_unmixing': self.ica_unmixing,
                'eps': self.eps
            }, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return cls(
            mean=data['mean'],
            pca_components=data['pca_components'],
            pca_explained_var=data['pca_explained_var'],
            ica_unmixing=data['ica_unmixing'],
            eps=data['eps']
        )



def encode_and_whiten_pcaica(sentences, st_model, whitening_model) -> np.ndarray:
    """
    Encode sentences with SentenceTransformer and whiten embeddings with PCA whitening model.

    Args:
        sentences (list of str or np.ndarray): List of sentences or numpy array of embeddings.
        st_model (SentenceTransformer): An initialized SentenceTransformer model.
        whitening_model (PCAICAWhiteningModel): An initialized PCAICAWhiteningModel.

    Returns:
        np.ndarray: Whitened embeddings.
    """
    if isinstance(sentences[0], str):
        # Step 1: Encode sentences to embeddings (numpy array)
        embeddings = st_model.encode(sentences, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    else:
        embeddings = sentences

    # Step 2: Apply whitening transform
    whitened_embeddings = whitening_model.transform(embeddings)

    return whitened_embeddings