import json
import numpy as np
from typing import Any

from src.benchmarks.base import BaseBenchmark
from src.benchmarks.msmarco import MSMarcoBenchmark
from src.util.benchmark_utils import (
    generate_unique_id, load_cobweb_model, print_metrics_table
)


def run_visualize_msmarco(model_name: str = "all-roberta-large-v1", subset_size: int = 7500,
                          split: str = "validation", target_size: int = 750, top_k: int = 3,
                          compute: bool = True, method: str = 'base', first_method: str = 'all',
                          second_method: str = 'all', transition_depth: int = 5,
                          target_dim: float = 0.96, num_leaves: int = 4, **kwargs: Any):
    """
    Run the MS Marco benchmark using the consolidated `BaseBenchmark` flow
    and then visualize the saved ApproxCobwebWrapper (PCA+ICA variant).
    """
    benchmark = MSMarcoBenchmark()

    # Recreate unique id used for caching so we can load the same Cobweb model
    unique_id = generate_unique_id(
        model_name=model_name,
        dataset="msmarco",
        split=split,
        subset_size=subset_size,
        target_size=target_size,
        top_k=top_k,
        method=method,
        first_method=first_method,
        second_method=second_method,
        transition_depth=transition_depth,
        target_dim=target_dim
    )

    # Recompute dataset, embeddings and PCA+ICA, then build a fresh ApproxCobwebWrapper
    print("Loading dataset and recomputing embeddings to build a fresh Cobweb...")
    # Load dataset (this will mirror BaseBenchmark behavior)
    corpus, queries, targets = benchmark.load_dataset(subset_size, split, target_size)

    # Detect DPR-style models
    is_dpr_model = "dpr-" in model_name and ("question_encoder" in model_name or "ctx_encoder" in model_name)

    # Setup embeddings (this will compute or load embeddings according to `compute` and unique_id)
    embeddings = benchmark.setup_embeddings(corpus, queries, targets, model_name, split, compute, unique_id, is_dpr_model=is_dpr_model)

    # Fit PCA+ICA and transform embeddings
    pca_ica_data = benchmark.setup_pca_ica_models(
        embeddings['corpus_embs'], embeddings['queries_embs'],
        model_name, split, unique_id, target_dim=target_dim, compute=compute
    )

    # Build a fresh Cobweb using PCA+ICA corpus embeddings (force compute)
    cobweb_pca_ica = load_cobweb_model(
        model_name, embeddings['corpus'], pca_ica_data['pca_ica_corpus_embs'],
        split, "pca_ica", unique_id=unique_id,
        first_method=first_method, second_method=second_method,
        transition_depth=transition_depth, force_compute=True
    )

    cobweb_pca_ica.tree.analyze_structure()

    print("Visualizing newly-built Cobweb PCA+ICA subtrees...")
    cobweb_pca_ica.visualize_query_subtrees(pca_ica_data['pca_ica_queries_embs'], queries, "outputs/visualizations_ms_marco", k=num_leaves)

def parse_args():
    parser = BaseBenchmark.create_argument_parser("Run MS Marco visualization with configurable parameters")
    parser.add_argument("--num_leaves", type=int, default=4, help="Number of leaves per visualization cluster")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Handle config file if provided
    args = BaseBenchmark.handle_config_and_args(args)

    # Print run info
    BaseBenchmark.print_run_info(args, "ms_marco")

    compute = args.compute

    results = run_visualize_msmarco(
        model_name=args.model_name,
        subset_size=args.subset_size,
        split=args.split,
        target_size=args.target_size,
        top_k=args.top_k,
        compute=compute,
        method=getattr(args, 'method', 'base'),
        first_method=getattr(args, 'first_method', 'all'),
        second_method=getattr(args, 'second_method', 'all'),
        transition_depth=getattr(args, 'transition_depth', 5),
        num_leaves=getattr(args, 'top_k', 4)
    )