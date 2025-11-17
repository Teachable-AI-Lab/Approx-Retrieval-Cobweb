# CUDA-accelerable PCA+ICA whitening using PyTorch
# Author: ChatGPT (adaptation of original numpy/scikit-learn version)

import torch
import pickle
from torch import nn


class PCAICAWhiteningModelTorch(nn.Module):
    def __init__(self, mean, pca_components, ica_unmixing, pca_explained_var, eps=1e-8):
        super().__init__()
        # Register buffers so they stay on the correct device and save/load easily.
        self.register_buffer("mean", mean)
        self.register_buffer("pca_components", pca_components)
        self.register_buffer("pca_explained_var", pca_explained_var)
        self.register_buffer("ica_unmixing", ica_unmixing)
        self.eps = eps

    def transform(self, x, is_ica=True):
        # x: torch.Tensor [n,d] or [d]
        is_single = (x.dim() == 1)
        if is_single:
            x = x.unsqueeze(0)

        # Step 1: center
        x_centered = x - self.mean

        # Step 2: PCA projection
        x_pca = torch.matmul(x_centered, self.pca_components.t())
        x_pca = x_pca / torch.sqrt(self.pca_explained_var + self.eps)

        # Step 3: ICA transform
        if is_ica:
            x_ica = torch.matmul(x_pca, self.ica_unmixing.t())
            return x_ica[0] if is_single else x_ica
        else:
            return x_pca[0] if is_single else x_pca

    @classmethod
    def fit(cls, X, pca_dim=256, eps=1e-8, ica_max_iter=5000, ica_tol=1e-3, device="cuda"):
        """
        Fit PCA → ICA whitening using Torch-based PCA and Torch-based ICA (fixed-point).
        """
        X = X.to(device)
        mean = X.mean(dim=0, keepdim=True)
        X_centered = X - mean

        # Step 1: PCA (via SVD)
        # X_centered: [N,D]
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        components = Vt[:pca_dim]  # [pca_dim, D]
        explained_var = (S[:pca_dim] ** 2) / (X.shape[0] - 1)

        # PCA projection
        X_pca = torch.matmul(X_centered, components.t())
        X_pca_normalized = X_pca / torch.sqrt(explained_var + eps)

        # Step 2: ICA (FastICA-like fixed-point iteration)
        n_comp = pca_dim
        W = torch.randn(n_comp, n_comp, device=device)
        W = torch.linalg.qr(W).Q

        for _ in range(ica_max_iter):
            WX = torch.matmul(W, X_pca_normalized.t())  # [k, N]
            g = torch.tanh(WX)
            g_prime = 1 - g ** 2

            W_new = (g @ X_pca_normalized) / X_pca_normalized.shape[0] - torch.diag(g_prime.mean(dim=1)) @ W

            # Symmetric orthogonalization
            W_new = torch.linalg.qr(W_new).Q

            if torch.max(torch.abs(torch.abs(torch.diag(W_new @ W.t())) - 1)) < ica_tol:
                W = W_new
                break
            W = W_new

        return cls(mean.squeeze(0), components, W, explained_var, eps)

    def save(self, filepath):
        data = {
            'mean': self.mean.detach().cpu(),
            'pca_components': self.pca_components.detach().cpu(),
            'pca_explained_var': self.pca_explained_var.detach().cpu(),
            'ica_unmixing': self.ica_unmixing.detach().cpu(),
            'eps': self.eps
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath, device="cuda"):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return cls(
            mean=data['mean'].to(device),
            pca_components=data['pca_components'].to(device),
            pca_explained_var=data['pca_explained_var'].to(device),
            ica_unmixing=data['ica_unmixing'].to(device),
            eps=data['eps']
        )

def encode_and_whiten_pcaica_torch(sentences, st_model, whitening_model, device="cuda"):
    if isinstance(sentences[0], str):
        embeddings = st_model.encode(sentences, convert_to_numpy=False)  # returns torch tensor for many ST versions
        embeddings = embeddings.to(device)
    else:
        embeddings = torch.tensor(sentences, device=device, dtype=torch.float32)

    whitened = whitening_model.transform(embeddings)
    return whitened
