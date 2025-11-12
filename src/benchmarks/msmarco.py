
from random import sample as randsample
from datasets import load_dataset
from typing import List, Tuple

from src.benchmarks.base import BaseBenchmark

class MSMarcoBenchmark(BaseBenchmark):
    """MS Marco dataset benchmark implementation."""
    
    def __init__(self):
        super().__init__("msmarco")
    
    def load_dataset(self, subset_size: int = 7500, split: str = "validation", 
                    target_size: int = 750) -> Tuple[List[str], List[str], List[str]]:
        """Load MS Marco dataset and extract query-target pairs."""
        # Load MS Marco dataset and extract query-target pairs
        ds = load_dataset("ms_marco", "v2.1", split=split)
        ds.shuffle()
        
        all_passages = []
        positive_pairs = []
        corpus = []
        
        for ex in ds:
            query = ex["query"]
            passage_texts = ex["passages"]["passage_text"]
            is_selected_flags = ex["passages"]["is_selected"]
            
            if any(is_selected_flags) and len(positive_pairs) < target_size:
                positive_pairs.append((query, passage_texts[is_selected_flags.index(1)]))
                corpus.extend(passage_texts)
            elif len(corpus) < subset_size:
                all_passages.extend(passage_texts)
            else:
                break
        
        if len(corpus) < subset_size:
            corpus.extend(randsample(all_passages, subset_size - len(corpus)))
        
        queries = [pair[0] for pair in positive_pairs]
        targets = [pair[1] for pair in positive_pairs]
        
        print(f"Loaded {len(corpus)} passages, {len(queries)} queries, and {len(targets)} targets.")
        
        return corpus, queries, targets


def run_msmarco_benchmark(model_name: str = "facebook/dpr-question_encoder-single-nq-base", 
                         subset_size: int = 7500, split: str = "validation", 
                         target_size: int = 750, top_k: int = 3, compute: bool = True, 
                         method: str = "all", first_method: str = 'all', second_method: str = 'all', transition_depth: int = 5) -> List[dict]:
    """Run MS Marco benchmark using the new consolidated approach."""
    benchmark = MSMarcoBenchmark()
    return benchmark.run_benchmark(
        model_name=model_name,
        subset_size=subset_size,
        split=split,
        target_size=target_size,
        top_k=top_k,
        compute=compute,
        method=method,
        target_dim=0.96,  # MS Marco uses 0.96 target dimension
        first_method=first_method,
        second_method=second_method,
        transition_depth=transition_depth
    )


def parse_args():
    """Parse command line arguments for MS Marco benchmark."""
    parser = BaseBenchmark.create_argument_parser("Run MS Marco Benchmark with configurable parameters")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Handle config file if provided
    args = BaseBenchmark.handle_config_and_args(args)
    
    # Print run information
    BaseBenchmark.print_run_info(args, "ms_marco")
    
    results = run_msmarco_benchmark(
        model_name=args.model_name,
        subset_size=args.subset_size,
        split=args.split,
        target_size=args.target_size,
        top_k=args.top_k,
        compute=args.compute,
        method=args.method,
        first_method=getattr(args, 'first_method', 'all'),
        second_method=getattr(args, 'second_method', 'all'),
        transition_depth=getattr(args, 'transition_depth', 5)
    )