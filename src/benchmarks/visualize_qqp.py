"""
Visualize QQP Cobweb subtrees using the consolidated benchmark flow.

This mirrors `visualize_msmarco.py` but uses the QQP dataset and `QQPBenchmark`.
"""

from typing import Any

from src.benchmarks.base import BaseBenchmark
from src.benchmarks.qqp import QQPBenchmark
from src.util.benchmark_utils import (
    generate_unique_id, load_cobweb_model, print_metrics_table
)


def run_visualize_qqp(model_name: str = "all-roberta-large-v1", subset_size: int = 7500,
                      split: str = "validation", target_size: int = 750, top_k: int = 3,
                      compute: bool = True, method: str = 'base', first_method: str = 'all',
                      second_method: str = 'all', transition_depth: int = 5,
                      target_dim: float = 0.90, num_leaves: int = 4, **kwargs: Any):
    """
    Run the QQP benchmark flow and visualize the saved ApproxCobwebWrapper (PCA+ICA variant).
    """
    benchmark = QQPBenchmark()

    unique_id = generate_unique_id(
        model_name=model_name,
        dataset="qqp",
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

    print("Loading dataset and recomputing embeddings to build a fresh Cobweb...")
    corpus, queries, targets = benchmark.load_dataset(subset_size, split, target_size)

    is_dpr_model = "dpr-" in model_name and ("question_encoder" in model_name or "ctx_encoder" in model_name)

    embeddings = benchmark.setup_embeddings(corpus, queries, targets, model_name, split, compute, unique_id, is_dpr_model=is_dpr_model)

    pca_ica_data = benchmark.setup_pca_ica_models(
        embeddings['corpus_embs'], embeddings['queries_embs'],
        model_name, split, unique_id, target_dim=target_dim, compute=compute
    )

    cobweb_pca_ica = load_cobweb_model(
        model_name, embeddings['corpus'], pca_ica_data['pca_ica_corpus_embs'],
        split, "pca_ica", unique_id=unique_id,
        first_method=first_method, second_method=second_method,
        transition_depth=transition_depth, force_compute=True
    )

    cobweb_pca_ica.tree.analyze_structure()

    print("Visualizing newly-built Cobweb PCA+ICA subtrees...")
    cobweb_pca_ica.visualize_query_subtrees(pca_ica_data['pca_ica_queries_embs'], queries, "outputs/visualizations_qqp", k=num_leaves)


def parse_args():
    parser = BaseBenchmark.create_argument_parser("Run QQP visualization with configurable parameters")
    parser.add_argument("--num_leaves", type=int, default=4, help="Number of leaves per visualization cluster")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args = BaseBenchmark.handle_config_and_args(args)

    BaseBenchmark.print_run_info(args, "qqp")

    compute = args.compute

    run_visualize_qqp(
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
        num_leaves=getattr(args, 'num_leaves', 4),
        target_dim=getattr(args, 'target_dim', 0.90)
    )