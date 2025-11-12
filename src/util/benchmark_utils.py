"""
Utilities for benchmarking different retrieval methods.
Provides common functionality for embeddings, model loading, evaluation, and file naming.
"""

import os
import numpy as np
import time
import json
import hashlib
from tqdm import tqdm
from tabulate import tabulate
from typing import List, Dict, Any, Tuple, Optional, Callable
from multiprocessing import Pool, cpu_count
from functools import partial

from transformers import (
    AutoTokenizer, T5EncoderModel, AutoModel,
    DPRQuestionEncoder, DPRContextEncoder, 
    DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
)
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score

import faiss
from annoy import AnnoyIndex
import hnswlib

from src.cobweb.ApproxCobwebWrapper import ApproxCobwebWrapper
from src.util.pca_ica import PCAICAWhiteningModel as PCAICAWhitening

# Model type mapping dictionary
MODEL_TYPE_MAPPING = {
    # Sentence Transformers
    "all-roberta-large-v1": {"type": "sentence_transformer"},
    "gtr-t5-large": {"type": "sentence_transformer"},
    
    # T5 models
    "google/t5-base": {"type": "transformer", "model_class": T5EncoderModel, "pooling": "mean"},
    
    # BERT models
    "bert-base-uncased": {"type": "transformer", "model_class": AutoModel, "pooling": "cls"},
    "bert-large-uncased": {"type": "transformer", "model_class": AutoModel, "pooling": "cls"},
    "SupstarZh/whitenedcse-bert-large": {
        "type": "transformer-repo",
        "model_class": AutoModel,
        "repoid": "unsup-whitenedcse-bert-large",
        "pooling": "cls"
    },
    # GPT2 models
    "openai-community/gpt2": {"type": "transformer", "model_class": AutoModel, "pooling": "mean", "add_pad_token": True},
}


class TextDataset(Dataset):
    """Simple dataset for text embeddings with GPU support."""
    def __init__(self, texts: List[str]):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]


def get_model_config(model_name: str) -> dict:
    """
    Get model configuration based on the model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dictionary
    """
    # Check exact matches first
    if model_name in MODEL_TYPE_MAPPING:
        return MODEL_TYPE_MAPPING[model_name]
    
    # Check partial matches for common patterns - order matters!
    model_lower = model_name.lower()
    
    # Check for more specific patterns first to avoid false matches
    if "roberta" in model_lower or "gtr" in model_lower:
        return {"type": "sentence_transformer"}
    elif "t5" in model_lower and "gtr" not in model_lower:  # Exclude gtr-t5 models
        return {"type": "transformer", "model_class": T5EncoderModel, "pooling": "mean"}
    elif "bert" in model_lower and "roberta" not in model_lower:  # Exclude roberta models
        return {"type": "transformer", "model_class": AutoModel, "pooling": "cls"}
    elif "gpt2" in model_lower:
        return {"type": "transformer", "model_class": AutoModel, "pooling": "mean", "add_pad_token": True}
    elif "whitenedcse" in model_lower:
        return {"type": "transformer-repo", "model_class": AutoModel, "repoid": "unsup-whitenedcse-bert-large", "pooling": "cls"}
    else:
        # Default to sentence transformer for unknown models
        return {"type": "sentence_transformer"}


def generate_unique_id(model_name: str, dataset: str, split: str, subset_size: int, 
                      target_size: int, **kwargs) -> str:
    """
    Generate a unique identifier based on all benchmark parameters.
    
    Args:
        model_name: Name of the model used
        dataset: Dataset name
        split: Data split (train/validation/test)
        subset_size: Size of the corpus subset
        target_size: Number of query-target pairs
        **kwargs: Additional parameters to include in the hash
    
    Returns:
        Unique identifier string
    """
    # Create a dictionary with all parameters
    params = {
        'model_name': model_name.replace('/', '-'),
        'dataset': dataset,
        'split': split,
        'subset_size': subset_size,
        'target_size': target_size,
        **kwargs
    }
    
    # Create a sorted string representation
    param_str = '_'.join([f"{k}={v}" for k, v in sorted(params.items())])
    
    # Generate a hash for very long parameter strings
    if len(param_str) > 100:
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        base_params = f"{params['model_name']}_{params['dataset']}_{params['split']}_c{params['subset_size']}_t{params['target_size']}_k{params['top_k']}"
        return f"{base_params}_{param_hash}"
    
    return param_str.replace('=', '').replace('/', '-')


def get_embedding_path(model_name: str, dataset: str, split: str, 
                      model_type: Optional[str] = None, unique_id: Optional[str] = None) -> str:
    """
    Generate path for embedding files with optional unique identifier.
    
    Args:
        model_name: Name of the model
        dataset: Dataset name
        split: Data split
        model_type: Optional model type (e.g., 'query', 'passage')
        unique_id: Optional unique identifier for the run
    
    Returns:
        Path to embedding file
    """
    os.makedirs("data/embeddings", exist_ok=True)
    model_name = model_name.replace('/', '-')
    
    if model_type and unique_id:
        return f"data/embeddings/{model_type}_{model_name}_{dataset}_{split}_{unique_id}.npy"
    elif model_type:
        return f"data/embeddings/{model_type}_{model_name}_{dataset}_{split}.npy"
    elif unique_id:
        return f"data/embeddings/{model_name}_{dataset}_{split}_{unique_id}.npy"
    else:
        return f"data/embeddings/{model_name}_{dataset}_{split}.npy"


def get_sentences_path(model_name: str, dataset: str, split: str, unique_id: Optional[str] = None) -> str:
    """Generate path for saved sentences with optional unique identifier."""
    os.makedirs("data/sentences", exist_ok=True)
    model_name = model_name.replace('/', '-')
    
    if unique_id:
        return f"data/sentences/{model_name}_{dataset}_{split}_{unique_id}.json"
    else:
        return f"data/sentences/{model_name}_{dataset}_{split}.json"


def get_model_path(model_name: str, split: str, mode: str, model_type: str, unique_id: Optional[str] = None) -> str:
    """
    Generate path for saved models with optional unique identifier.
    
    Args:
        model_name: Name of the model
        split: Data split
        mode: Model mode (e.g., 'base', 'pca_ica')
        model_type: Type of model (e.g., 'cobweb_wrappers', 'pca_ica')
        unique_id: Optional unique identifier
    
    Returns:
        Path to model file
    """
    os.makedirs(f"models/{model_type}", exist_ok=True)
    model_name = model_name.replace('/', '-')
    
    if unique_id:
        if model_type == 'cobweb_wrappers':
            return f"models/{model_type}/{model_name}_{split}_{mode}_{unique_id}.json"
        else:
            return f"models/{model_type}/{model_name}_{split}_{mode}_{unique_id}.pkl"
    else:
        if model_type == 'cobweb_wrappers':
            return f"models/{model_type}/{model_name}_{split}_{mode}.json"
        else:
            return f"models/{model_type}/{model_name}_{split}_{mode}.pkl"


def get_results_path(model_name: str, dataset: str, split: str, unique_id: str) -> str:
    """Generate path for benchmark results with unique identifier."""
    os.makedirs(f"outputs/{dataset}", exist_ok=True)
    model_name = model_name.replace('/', '-')
    return f"outputs/{dataset}/benchmark_{model_name}_{split}_{unique_id}.txt"


def _compute_embeddings_gpu(texts: List[str], tokenizer, model, model_config: dict, batch_size: int = 32) -> np.ndarray:
    """Compute embeddings using GPU with DataLoader for efficient batch processing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader with CPU workers
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=min(8, len(os.sched_getaffinity(0))))
    
    all_embeddings = []
    pooling = model_config["pooling"]
    
    with torch.no_grad():
        for batch_texts in tqdm(dataloader, desc="Computing embeddings"):
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, 
                             truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            last_hidden = outputs.hidden_states[-1]
            
            if pooling == "cls":
                batch_embeddings = last_hidden[:, 0, :].cpu()
            elif pooling == "mean":
                mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                sum_hidden = (last_hidden * mask).sum(dim=1)
                divisor = mask.sum(dim=1).clamp(min=1e-9)
                batch_embeddings = (sum_hidden / divisor).cpu()
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")
            
            all_embeddings.append(batch_embeddings.numpy())
    
    return np.vstack(all_embeddings)


def _compute_embeddings_cpu(texts: List[str], tokenizer, model, model_config: dict) -> np.ndarray:
    """Compute embeddings using CPU (original implementation)."""
    model.eval()
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    pooling = model_config["pooling"]
    if pooling == "cls":
        last_hidden = outputs.hidden_states[-1]
        embeddings = last_hidden[:, 0, :].numpy()
    elif pooling == "mean":
        last_hidden = outputs.hidden_states[-1]
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
        sum_hidden = (last_hidden * mask).sum(dim=1)
        divisor = mask.sum(dim=1).clamp(min=1e-9)
        embeddings = (sum_hidden / divisor).cpu().numpy()
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}")
    
    return embeddings


def load_or_compute_embeddings(texts: List[str], model_name: str, dataset: str, split: str, 
                             compute: bool = False, unique_id: Optional[str] = None) -> np.ndarray:
    """
    Load or compute embeddings using different model types.
    
    Args:
        texts: List of texts to encode
        model_name: Name of the model to use
        dataset: Dataset name
        split: Data split
        compute: Whether to force recomputation
        unique_id: Optional unique identifier for this run
    
    Returns:
        Numpy array of embeddings
    """
    path = get_embedding_path(model_name, dataset, split, unique_id=unique_id)
    
    if os.path.exists(path) and not compute:
        print(f"Loading embeddings from {path}")
        return np.load(path)
    
    print(f"Computing embeddings and saving to {path}")
    print(f"texts : {texts[:5]}...")  # Show first 5 texts for debugging
    
    model_config = get_model_config(model_name)
    print(f"Using model config: {model_config} for model: {model_name}")
    
    if model_config["type"] == "transformer":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if 't5' in model_name.lower():
            texts = ["Summarize :" + text for text in texts]
        
        # Handle GPT2 pad token
        if model_config.get("add_pad_token", False) and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model_class = model_config["model_class"]
        model = model_class.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)
        
        # Choose GPU or CPU computation based on availability
        print("Using GPU with CPU workers for embedding computation")
        embeddings = _compute_embeddings_gpu(texts, tokenizer, model, model_config)
    
    elif model_config["type"] == "transformer-repo":
        repo_id = model_name
        subfolder = model_config["repoid"]
        tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder, trust_remote_code=True)
        model_class = model_config["model_class"]
        model = model_class.from_pretrained(repo_id, subfolder=subfolder, trust_remote_code=True, output_hidden_states=True)

        embeddings = _compute_embeddings_gpu(texts, tokenizer, model, model_config)

    else:
        # print(f"Unknown model type {model_config['type']}, falling back to SentenceTransformer")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        model.to(device)
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True, batch_size=256)

    print(f"Computed embeddings shape: {embeddings.shape}")
    if len(embeddings.shape) != 2:
        raise ValueError(f"Embeddings for {model_name} should be 2D, got {embeddings.shape}. Check the model and input texts.")
    
    np.save(path, embeddings)
    return embeddings


def load_or_compute_dpr_embeddings(texts: List[str], model_type: str, model_name: str, 
                                 dataset: str, split: str, compute: bool = False, 
                                 unique_id: Optional[str] = None) -> np.ndarray:
    """
    Load or compute DPR embeddings for either queries or passages.
    
    Args:
        texts: List of texts to encode
        model_type: Either 'query' or 'passage'
        model_name: Base DPR model name
        dataset: Dataset name
        split: Data split
        compute: Whether to force recomputation
        unique_id: Optional unique identifier for this run
    
    Returns:
        Numpy array of embeddings
    """
    path = get_embedding_path(model_name, dataset, split, model_type=model_type, unique_id=unique_id)
    
    if os.path.exists(path) and not compute:
        print(f"Loading {model_type} embeddings from {path}")
        return np.load(path)
    else:
        print(f"Computing {model_type} embeddings and saving to {path}")
        print(f"texts : {texts[:5]}...")  # Show first 5 texts for debugging
        
        if model_type == 'query':
            tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
            model = DPRQuestionEncoder.from_pretrained(model_name)
        elif model_type == 'passage':
            tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
            model = DPRContextEncoder.from_pretrained(model_name)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Process in batches to avoid memory issues
        batch_size = 10
        embeddings = []
        model.eval()
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Computing {model_type} embeddings"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.pooler_output.numpy()
                embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        print(f"Computed {model_type} embeddings shape: {embeddings.shape}")
        
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings for {model_type} should be 2D, got {embeddings.shape}. Check the model and input texts.")
        
        np.save(path, embeddings)
        return embeddings


def load_or_save_sentences(sentences: Optional[List[str]], model_name: str, dataset: str, 
                          split: str, compute: bool = False, unique_id: Optional[str] = None) -> List[str]:
    """
    Load or save sentences to/from file.
    
    Args:
        sentences: List of sentences to save (None to load)
        model_name: Name of the model
        dataset: Dataset name
        split: Data split
        compute: Whether to force saving
        unique_id: Optional unique identifier for this run
    
    Returns:
        List of sentences
    """
    path = get_sentences_path(model_name, dataset, split, unique_id=unique_id)
    
    if sentences is None:
        print(f"Loading sentences from {path}")
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    else:
        print(f"Saving sentences to {path}")
        with open(path, 'w+') as f:
            f.write('\n'.join(sentences))
        return sentences


def load_cobweb_model(model_name: str, corpus: List[str], corpus_embs: np.ndarray, 
                     split: str, mode: str, unique_id: Optional[str] = None, 
                     first_method: str = 'bfs', second_method: str = 'pathsum',
                     transition_depth: int = 5, force_compute: bool = True) -> ApproxCobwebWrapper:
    """
    Load or create a Cobweb model.
    
    Args:
        model_name: Name of the model
        corpus: List of corpus texts
        corpus_embs: Corpus embeddings
        split: Data split
        mode: Model mode (e.g., 'base', 'pca_ica')
        unique_id: Optional unique identifier
        force_compute: Whether to force recomputation
    
    Returns:
        CobwebWrapper instance
    """
    # Include method configuration in the filename so different combos are cached separately
    method_tag = f"fm-{first_method}_sm-{second_method}_td-{transition_depth}"
    cobweb_path = get_model_path(model_name, split, mode, 'cobweb_wrappers', unique_id=f"{unique_id}_{method_tag}")

    if os.path.exists(cobweb_path) and not force_compute:
        print(f"Loading Cobweb model from {cobweb_path}")
        with open(cobweb_path, 'r') as f:
            cobweb_json = json.load(f)
        return ApproxCobwebWrapper.load_json(cobweb_json, encode_func=lambda x: x)
    else:
        print(f"Computing ApproxCobwebWrapper (first={first_method}, second={second_method}, td={transition_depth}) and saving to {cobweb_path}")
        cobweb = ApproxCobwebWrapper(first_method=first_method, second_method=second_method, transition_depth=transition_depth, corpus=corpus, corpus_embeddings=corpus_embs, encode_func=lambda x: x)
        try:
            cobweb.dump_json(cobweb_path)
        except Exception as e:
            print(f"Warning: failed to dump Cobweb model to {cobweb_path}: {e}")
        return cobweb


def load_pca_ica_model(corpus_embs: np.ndarray, model_name: str, dataset: str, split: str, 
                      model_type: str, target_dim: float = 0.96, unique_id: Optional[str] = None, compute: bool = False) -> PCAICAWhitening:
    """
    Load or create a PCA + ICA whitening model.
    
    Args:
        corpus_embs: Corpus embeddings to fit the model
        model_name: Name of the base model
        dataset: Dataset name
        split: Data split
        model_type: Model type identifier
        target_dim: Target dimensionality (as fraction or int)
        unique_id: Optional unique identifier
    
    Returns:
        PCAICAWhitening model
    """
    dim_str = '_'.join(str(target_dim).split('.'))
    mode = f"{model_type}_{dim_str}"
    pca_ica_path = get_model_path(model_name, split, mode, 'pca_ica', unique_id)
    
    if os.path.exists(pca_ica_path) and not compute:
        print(f"Loading PCA + ICA model for {model_type} from {pca_ica_path}")
        return PCAICAWhitening.load(pca_ica_path)
    else:
        print(f"Computing PCA + ICA model for {model_type} and saving to {pca_ica_path}")
        pca_ica_model = PCAICAWhitening.fit(corpus_embs, pca_dim=target_dim)
        pca_ica_model.save(pca_ica_path)
        return pca_ica_model


# === Retrieval Setup Functions ===

def setup_cobweb_basic(corpus: List[str], corpus_embs: np.ndarray) -> ApproxCobwebWrapper:
    """Setup a basic Cobweb wrapper."""
    return ApproxCobwebWrapper(corpus=corpus, corpus_embeddings=corpus_embs, encode_func=lambda x: x)


def setup_faiss(corpus_embs: np.ndarray, index_type: str = "dot"):
    """Setup a FAISS index for inner product similarity."""
    dim = corpus_embs.shape[1]
    if index_type == "dot":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unknown FAISS index type: {index_type}")
    index.add(corpus_embs)
    return index


def setup_faiss_ivf_pq(corpus_embs: np.ndarray, nlist: int = 100, m: int = 8, nbits: int = 8):
    """Setup a FAISS IVF-PQ index (approximate) using an L2 quantizer and train it on the corpus.

    This index is useful for approximate retrieval when exact FAISS (flat) is too heavy.
    """
    xb = corpus_embs.astype(np.float32)
    dim = xb.shape[1]

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

    if not index.is_trained:
        print(f"Training IVF-PQ (nlist={nlist}, m={m}, nbits={nbits}) on {xb.shape[0]} vectors...")
        index.train(xb)

    index.add(xb)
    return index


def setup_annoy(corpus_embs: np.ndarray) -> AnnoyIndex:
    """Setup an Annoy index for approximate nearest neighbor search."""
    dim = corpus_embs.shape[1]
    index = AnnoyIndex(dim, 'angular')
    for i, emb in enumerate(corpus_embs):
        index.add_item(i, emb)
    index.build(10)
    return index


def setup_hnswlib(corpus_embs: np.ndarray) -> hnswlib.Index:
    """Setup an HNSW index for approximate nearest neighbor search."""
    dim = corpus_embs.shape[1]
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=len(corpus_embs), ef_construction=100, M=16)
    index.add_items(corpus_embs, np.arange(len(corpus_embs)))
    index.set_ef(50)
    return index


def setup_torch_dot(corpus_embs: np.ndarray) -> torch.Tensor:
    """Setup torch tensor for direct dot product similarity computation."""
    return torch.from_numpy(corpus_embs).float()


# === Retrieval Functions ===

def retrieve_cobweb_basic(query_emb: np.ndarray, k: int, cobweb: ApproxCobwebWrapper, use_fast: bool = False) -> List[str]:
    """Retrieve using Cobweb wrapper."""
    if use_fast:
        return cobweb.cobweb_predict_fast(query_emb, k)
    else:
        return cobweb.cobweb_predict(query_emb, k)


def retrieve_faiss(query_emb: np.ndarray, k: int, index: faiss.IndexFlatIP, corpus: List[str]) -> List[str]:
    """Retrieve using FAISS index."""
    _, ids = index.search(np.expand_dims(query_emb, axis=0), k)
    return [corpus[i] for i in ids[0]]


def retrieve_annoy(query_emb: np.ndarray, k: int, index: AnnoyIndex, corpus: List[str]) -> List[str]:
    """Retrieve using Annoy index."""
    ids = index.get_nns_by_vector(query_emb, k)
    return [corpus[i] for i in ids]


def retrieve_hnswlib(query_emb: np.ndarray, k: int, index: hnswlib.Index, corpus: List[str]) -> List[str]:
    """Retrieve using HNSW index."""
    ids, _ = index.knn_query(query_emb, k=k)
    return [corpus[i] for i in ids[0]]


def retrieve_torch_dot(query_emb: np.ndarray, k: int, corpus_embs: torch.Tensor, corpus: List[str]) -> List[str]:
    """Retrieve using PyTorch matrix multiplication for dot product similarity."""
    # Convert query to torch tensor
    query_tensor = torch.from_numpy(query_emb).float()
    
    # Compute dot product similarities using matrix multiplication
    # corpus_embs is (num_docs, embedding_dim), query_tensor is (embedding_dim,)
    similarities = torch.matmul(corpus_embs, query_tensor)
    
    # Get top-k indices in descending order
    _, top_k_indices = torch.topk(similarities, k, largest=True)
    
    return [corpus[i.item()] for i in top_k_indices]


# === Evaluation Functions ===

def get_eval_ks(top_k: int) -> List[int]:
    """Return a list of k-values to evaluate based on top_k."""
    base = [2, 3, 5, 10, 20, 50, 100]
    return sorted([k for k in base if k <= top_k])


def _evaluate_single_query(args):
    """
    Helper function to evaluate a single query-target pair.
    This function is designed to be used with multiprocessing.
    
    Args:
        args: Tuple containing (query, target, retrieve_fn, top_k, ks)
    
    Returns:
        Tuple containing (query_metrics, latency, success, error_msg)
        If success=False, query_metrics and latency will be None
    """
    query, target, retrieve_fn, top_k, ks = args
    
    try:
        # Initialize metrics for this query
        query_metrics = {f"recall@{k}": 0 for k in ks}
        query_metrics.update({f"mrr@{k}": 0 for k in ks})
        query_metrics.update({f"ndcg@{k}": 0 for k in ks})
        
        # Measure retrieval time
        start = time.time()
        retrieved = retrieve_fn(query, top_k)
        latency = time.time() - start
        
        # Calculate metrics for each k
        for k in ks:
            top_k_results = retrieved[:k]
            if target in top_k_results:
                query_metrics[f"recall@{k}"] = 1
                rank = top_k_results.index(target) + 1
                query_metrics[f"mrr@{k}"] = 1 / rank
            
            relevance = [1 if doc == target else 0 for doc in top_k_results]
            if sum(relevance) > 0:
                ideal = sorted(relevance, reverse=True)
                ndcg = ndcg_score([ideal], [relevance])
                query_metrics[f"ndcg@{k}"] = ndcg
        
        return query_metrics, latency, True, None
        
    except Exception as e:
        # Return error information for fallback processing
        return None, None, False, str(e)


def _evaluate_single_query_sequential(query, target, retrieve_fn, top_k, ks):
    """
    Sequential fallback function to evaluate a single query-target pair.
    Used when parallel processing fails for specific queries.
    
    Args:
        query: Query embedding
        target: Target string (ground truth)
        retrieve_fn: Function to retrieve results
        top_k: Maximum number of results to retrieve
        ks: List of k values to evaluate
    
    Returns:
        Tuple containing (query_metrics, latency)
    """
    query_metrics = {f"recall@{k}": 0 for k in ks}
    query_metrics.update({f"mrr@{k}": 0 for k in ks})
    query_metrics.update({f"ndcg@{k}": 0 for k in ks})
    
    start = time.time()
    retrieved = retrieve_fn(query, top_k)
    latency = time.time() - start
    
    for k in ks:
        top_k_results = retrieved[:k]
        if target in top_k_results:
            query_metrics[f"recall@{k}"] = 1
            rank = top_k_results.index(target) + 1
            query_metrics[f"mrr@{k}"] = 1 / rank
        
        relevance = [1 if doc == target else 0 for doc in top_k_results]
        if sum(relevance) > 0:
            ideal = sorted(relevance, reverse=True)
            ndcg = ndcg_score([ideal], [relevance])
            query_metrics[f"ndcg@{k}"] = ndcg
    
    return query_metrics, latency


def evaluate_retrieval(name: str, queries: List[np.ndarray], targets: List[str], 
                      retrieve_fn: Callable, top_k: int = 10, use_parallel: bool = False) -> Dict[str, Any]:
    """
    Evaluate a retrieval method using multiple metrics with optional parallelization and error handling.
    
    Args:
        name: Name of the retrieval method
        queries: List of query embeddings
        targets: List of target strings (ground truth)
        retrieve_fn: Function to retrieve results given (query, k)
        top_k: Maximum number of results to retrieve
        use_parallel: Whether to use parallel processing (default: True)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    ks = get_eval_ks(top_k)
    metrics = {
        f"recall@{k}": 0 for k in ks
    }
    metrics.update({
        f"mrr@{k}": 0 for k in ks
    })
    metrics.update({
        f"ndcg@{k}": 0 for k in ks
    })

    if use_parallel:
        # Prepare arguments for parallel processing
        args_list = [(query, target, retrieve_fn, top_k, ks) for query, target in zip(queries, targets)]
        
        # Use all available CPU cores
        num_workers = cpu_count()
        print(f"Using {num_workers} parallel workers for evaluation")
        
        try:
            with Pool(processes=num_workers) as pool:
                # Process all queries in parallel with progress bar
                results = list(tqdm(
                    pool.imap(_evaluate_single_query, args_list), 
                    total=len(queries), 
                    desc=f"Evaluating {name} (parallel)"
                ))
        except Exception as e:
            print(f"Parallel processing failed entirely: {e}")
            print("Falling back to sequential processing for all queries")
            use_parallel = False
        
        if use_parallel:  # Only process results if parallel processing succeeded
            # Separate successful and failed results
            successful_results = []
            failed_queries = []
            latencies = []
            
            for i, (query_metrics, latency, success, error_msg) in enumerate(results):
                if success:
                    successful_results.append((query_metrics, latency))
                    latencies.append(latency)
                    # Add to metrics
                    for k in ks:
                        metrics[f"recall@{k}"] += query_metrics[f"recall@{k}"]
                        metrics[f"mrr@{k}"] += query_metrics[f"mrr@{k}"]
                        metrics[f"ndcg@{k}"] += query_metrics[f"ndcg@{k}"]
                else:
                    failed_queries.append((i, queries[i], targets[i], error_msg))
            
            # Process failed queries sequentially
            if failed_queries:
                print(f"Processing {len(failed_queries)} failed queries sequentially...")
                for idx, query, target, error_msg in tqdm(failed_queries, desc="Processing failed queries"):
                    try:
                        query_metrics, latency = _evaluate_single_query_sequential(
                            query, target, retrieve_fn, top_k, ks
                        )
                        latencies.append(latency)
                        # Add to metrics
                        for k in ks:
                            metrics[f"recall@{k}"] += query_metrics[f"recall@{k}"]
                            metrics[f"mrr@{k}"] += query_metrics[f"mrr@{k}"]
                            metrics[f"ndcg@{k}"] += query_metrics[f"ndcg@{k}"]
                    except Exception as fallback_error:
                        print(f"Query {idx} failed even in sequential mode: {fallback_error}")
                        # Add zero metrics and default latency for completely failed queries
                        latencies.append(0.0)
            
            print(f"Successfully processed {len(successful_results)} queries in parallel, "
                  f"{len(failed_queries)} queries sequentially")
                
    if not use_parallel:
        # Original sequential processing (fallback for complete parallel failure)
        latencies = []
        for query, target in tqdm(zip(queries, targets), total=len(queries), desc=f"Evaluating {name} (sequential)"):
            try:
                start = time.time()
                retrieved = retrieve_fn(query, top_k)
                latencies.append(time.time() - start)

                for k in ks:
                    top_k_results = retrieved[:k]
                    if target in top_k_results:
                        metrics[f"recall@{k}"] += 1
                        rank = top_k_results.index(target) + 1
                        metrics[f"mrr@{k}"] += 1 / rank
                    relevance = [1 if doc == target else 0 for doc in top_k_results]
                    if sum(relevance) > 0:
                        ideal = sorted(relevance, reverse=True)
                        ndcg = ndcg_score([ideal], [relevance])
                        metrics[f"ndcg@{k}"] += ndcg
            except Exception as e:
                print(f"Query failed in sequential mode: {e}")
                latencies.append(0.0)

    # Normalize metrics by number of queries
    n = len(queries)
    for k in ks:
        metrics[f"recall@{k}"] = round(metrics[f"recall@{k}"] / n, 4)
        metrics[f"mrr@{k}"] = round(metrics[f"mrr@{k}"] / n, 4)
        metrics[f"ndcg@{k}"] = round(metrics[f"ndcg@{k}"] / n, 4)
    
    metrics["time_taken"] = round(np.sum(latencies), 2)
    metrics["method"] = name
    metrics["avg_latency_ms"] = round(1000 * np.mean(latencies), 2)

    return metrics


def print_metrics_table(metrics: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Print and optionally save metrics in a formatted table.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        save_path: Optional path to save the results
    """
    method = metrics.pop("method", "Unknown")
    latency = metrics.pop("avg_latency_ms", None)
    total_time = metrics.pop("time_taken", 0)

    rows = []
    ks = sorted(set(int(k.split('@')[1]) for k in metrics if '@' in k))
    for k in ks:
        row = [
            f"@{k}",
            metrics.get(f"recall@{k}", 0),
            metrics.get(f"mrr@{k}", 0),
            metrics.get(f"ndcg@{k}", 0)
        ]
        rows.append(row)

    table_str = f"\n--- Metrics for {method} ---\n"
    if latency is not None:
        table_str += f"Avg Latency: {latency} ms with total time {total_time} seconds\n"
    table_str += tabulate(rows, headers=["k", "Recall", "MRR", "nDCG"], tablefmt="pretty")

    print(table_str)

    if save_path:
        with open(save_path, "a+") as f: 
            f.write(table_str + "\n")