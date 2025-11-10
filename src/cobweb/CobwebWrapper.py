import torch
import json
import random
import math
from tqdm import tqdm
from collections import deque
import os
import hashlib
from graphviz import Digraph
from src.cobweb.CobwebTorchTree import CobwebTorchTree

class CobwebWrapper:
    def __init__(self, corpus=None, corpus_embeddings=None, encode_func=lambda x: x):
        """
        Initializes the CobwebWrapper with optional sentences and/or embeddings.
        """

        self.encode_func = encode_func

        self.sentences = []
        self.sentence_to_node = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_init_search = 100000

        # Prediction index caching
        self._prediction_index_valid = False
        self._index_to_node = {}
        self._node_means = None
        self._node_vars = None
        self._leaf_to_path_indices = None
        self.max_depth = 0

        # Determine embedding shape
        if corpus_embeddings is not None:
            corpus_embeddings = torch.tensor(corpus_embeddings) if isinstance(corpus_embeddings, list) else corpus_embeddings
            embedding_shape = corpus_embeddings.shape[1:]
        elif corpus and len(corpus) > 0:
            sample_emb = self.encode_func([corpus[0]])
            embedding_shape = sample_emb.shape[1:]


        self.tree = CobwebTorchTree(shape=embedding_shape, device=self.device)

        if corpus_embeddings is not None:
            if corpus is None:
                corpus = [None] * len(corpus_embeddings)
            self.add_sentences(corpus, corpus_embeddings)
        elif corpus is not None and len(corpus) > 0:
            self.add_sentences(corpus)

    def add_sentences(self, new_sentences, new_vectors=None):
        """
        Adds new sentences and/or embeddings to the Cobweb tree.
        If a sentence is None, it is treated as an embedding-only entry.
        """
        if new_vectors is None:
            new_embeddings = self.encode_func(new_sentences)
        else:
            new_embeddings = new_vectors
            if isinstance(new_embeddings, list):
                new_embeddings = torch.tensor(new_embeddings)
            if new_embeddings.shape[1] != self.tree.shape[0]:
                print(f"[Warning] Provided vector dim {new_embeddings.shape[1]} != tree dim {self.tree.shape[0]}, re-encoding...")
                new_embeddings = self.encode_func(new_sentences)

        start_index = len(self.sentences)

        for i, (sent, emb) in tqdm(enumerate(zip(new_sentences, new_embeddings)),
                                   total=len(new_sentences),
                                   desc="Training CobwebTree"):
            self.sentences.append(sent)
            leaf = self.tree.ifit(torch.tensor(emb, device=self.device))
            if leaf.sentence_id is None:
                leaf.sentence_id = []
            leaf.sentence_id.append(start_index + i)
            self.sentence_to_node[start_index + i] = leaf

        # Invalidate prediction index when new sentences are added
        self._invalidate_prediction_index()

    def _invalidate_prediction_index(self):
        """Invalidate the prediction index when tree structure changes"""
        self._prediction_index_valid = False
        self._index_to_node.clear()
        self._node_means = None
        self._node_vars = None
        self._leaf_to_path_indices = None
        self._path_matrix = None

    def build_prediction_index(self):
        """
        Build an index of all nodes in the tree for faster prediction.
        Creates mappings between nodes and indices, and caches means/variances.
        """
        if self._prediction_index_valid:
            return
        print("Building prediction index...")

        if set(self.sentence_to_node.keys()) != set(range(len(self.sentences))):
            raise ValueError("sentence_to_node mapping is inconsistent with sentence indices.")
        
        # Clear existing mappings
        self._index_to_node.clear()
        new_sentences = [None] * len(self.sentences)

        # Collect all nodes via BFS traversal
        idx = 0
        leaf_idx = 0
        queue = [(self.tree.root, tuple())]
        self._leaf_to_path_indices = [None] * len(self.sentences)
        new_sentence_to_node = {}
        while queue:
            node, path = queue[0]
            queue = queue[1:]
            self._index_to_node[idx] = node
            for child in getattr(node, 'children', []):
                queue.append((child, path + (idx,)))
            if hasattr(node, 'sentence_id') and node.sentence_id:
                for sid in node.sentence_id:
                    if sid < len(self.sentences):
                        self._leaf_to_path_indices[sid] = list(path)+[idx]
                        new_sentence_to_node[sid] = node
                        new_sentences[sid] = self.sentences[sid]
                    else:
                        print(f"[Warning] Node has invalid sentence ID {sid}, skipping.")
                self.max_depth = max(self.max_depth, len(path) + 1)
                # new_sentence_to_node[leaf_idx] = node
                # new_sentences[leaf_idx] = self.sentences[node.sentence_id]
                # node.sentence_id = leaf_idx
                leaf_idx += 1
            idx += 1
        # self.sentence_to_node = new_sentence_to_node
        # self.sentences = new_sentences
        if leaf_idx != len(self.sentences):
            print(f"[Warning] Leaf count mismatch: expected {len(self.sentences)}, found {leaf_idx}.")
        for i, sid in enumerate(self._leaf_to_path_indices):
            if sid is None:
                print(f"[Warning] Leaf path index for sentence ID {i} is None. This may indicate missing sentences in the tree.")
                node = self.sentence_to_node.get(i, None)
                print(node.sentence_id, i)
            if node not in self._index_to_node.values():
                print(f"[Warning] Node for sentence ID {i} not found in indexed nodes. This may indicate a bug.")

        # Build sparse path matrix for efficient path scoring
        num_leaves = len(self._leaf_to_path_indices)
        num_nodes = idx
        path_row_indices = []
        path_col_indices = []
        path_weights = []
        
        # Default level weights - can be customized
        if not hasattr(self, '_level_weights') or self._level_weights is None:
            # Default to constant schedule with value 1.0
            level_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        else:
            level_weights = self._level_weights
    
        
        for leaf_idx, path in enumerate(self._leaf_to_path_indices):
            path_length = len(path)
            for depth, node_idx in enumerate(path):
                path_row_indices.append(leaf_idx)
                path_col_indices.append(node_idx)
                # Get weight for this level, default to 1.0 if beyond specified weights
                weight = level_weights[depth] if depth < len(level_weights) else 1.0
                # Normalize by path length to give equal total weight to all paths
                normalized_weight = weight / path_length
                path_weights.append(normalized_weight)
        
        # Create sparse matrix: [num_leaves, num_nodes]
        if path_row_indices:  # Only create if we have paths
            path_indices = torch.stack([
                torch.tensor(path_row_indices, device=self.device),
                torch.tensor(path_col_indices, device=self.device)
            ])
            path_values = torch.tensor(path_weights, device=self.device, dtype=torch.float)
            self._path_matrix = torch.sparse_coo_tensor(
                path_indices, path_values, 
                (num_leaves, num_nodes), 
                device=self.device
            ).coalesce()
        else:
            self._path_matrix = None

        # Pre-allocate tensors for means and variances
        num_nodes = idx
        self._node_means = torch.zeros(num_nodes, self.tree.shape[0], 
                                     device=self.device, dtype=torch.float)
        self._node_vars = torch.zeros(num_nodes, self.tree.shape[0], 
                                    device=self.device, dtype=torch.float)



        # Fill tensors with node statistics
        for idx, node in self._index_to_node.items():
            self._node_means[idx] = node.mean
            # Compute variance using the tree's variance computation method
            if hasattr(node, 'meanSq') and node.count > 0:
                self._node_vars[idx] = self.tree.compute_var(node.meanSq, node.count)
            else:
                # Use prior variance for empty nodes
                self._node_vars[idx] = self.tree.prior_var

        self._prediction_index_valid = True
        print(f"Prediction index built: {num_nodes} nodes indexed, {leaf_idx} leaf paths cached")
        if self._path_matrix is not None:
            print(f"Path matrix shape: {self._path_matrix.shape}, nnz: {self._path_matrix._nnz()}")

    def cobweb_predict_indexed(self, input, k=5, return_ids=False, is_embedding=False):
        """
        Ultra-fast prediction using sparse matrix operations for path scoring.
        """
        # Ensure prediction index is built
        self.build_prediction_index()
        
        if is_embedding:
            emb = input
        else:
            emb = self.encode_func([input])[0]

        x = torch.tensor(emb, device=self.device)  # (D,)
        
        num_leaves = len(self._leaf_to_path_indices)
        if num_leaves == 0:
            return []
        
        # Compute log probabilities for all nodes at once
        # log_2pi = math.log(2 * math.pi)
        diff_sq = (x.unsqueeze(0) - self._node_means) ** 2  # (num_nodes, D)
        
        node_log_probs = -0.5 * (
            # log_2pi * self.tree.shape[0] +
            torch.log(self._node_vars).sum(dim=1) +
            (diff_sq / self._node_vars).sum(dim=1)
        )  # (num_nodes,)
        
        # Sum log probabilities along each leaf's path using sparse matrix multiplication
        # self._path_matrix is [num_leaves, num_nodes], node_log_probs is [num_nodes]
        if self._path_matrix is not None:
            leaf_scores = torch.sparse.mm(self._path_matrix, node_log_probs.unsqueeze(1)).squeeze(1)  # (num_leaves,)
        else:
            return []
        
        # Get top-k results
        if k >= num_leaves:
            # Add small random noise to break ties randomly
            noise = torch.randn_like(leaf_scores) * 1e-6
            noisy_scores = leaf_scores + noise
            sorted_indices = torch.argsort(noisy_scores, descending=True)
            selected_leaf_indices = sorted_indices.tolist()
        else:
            # Add small random noise to break ties randomly
            noise = torch.randn_like(leaf_scores) * 1e-6
            noisy_scores = leaf_scores + noise
            _, topk_indices = torch.topk(noisy_scores, k, largest=True)
            selected_leaf_indices = topk_indices.tolist()
        
        # Convert leaf indices to sentence IDs and results
        results = []
        for leaf_idx in selected_leaf_indices:
            if leaf_idx < len(self.sentences):
                results.append(leaf_idx if return_ids else self.sentences[leaf_idx])
        
        return results

    def cobweb_rank_scores(self, input, is_embedding=False):
        """
        Differentiable: return raw scores for all leaves.
        """
        self.build_prediction_index()

        if is_embedding:
            x = input.to(self.device)  # already a torch tensor
        else:
            emb = self.encode_func([input])[0]
            x = torch.tensor(emb, device=self.device)

        if len(self._leaf_to_path_indices) == 0:
            return torch.empty(0, device=self.device)

        # Gaussian log-probs
        diff_sq = (x.unsqueeze(0) - self._node_means) ** 2
        node_log_probs = -0.5 * (
            torch.log(self._node_vars).sum(dim=1)
            + (diff_sq / self._node_vars).sum(dim=1)
        )  # (num_nodes,)

        # Aggregate along leaf paths
        leaf_scores = torch.sparse.mm(
            self._path_matrix, node_log_probs.unsqueeze(1)
        ).squeeze(1)  # (num_leaves,)

        return leaf_scores  # no top-k cut, differentiable


    def get_node_path_stats(self, sentence_id):
        """
        Get statistics for all nodes in the path from root to a specific leaf.
        Returns means and variances for the entire path.
        """
        self.build_prediction_index()
        
        if sentence_id not in self._leaf_to_path_indices:
            return None, None
            
        path_indices = self._leaf_to_path_indices[sentence_id]
        path_indices_tensor = torch.tensor(path_indices, device=self.device)
        
        path_means = self._node_means[path_indices_tensor]
        path_vars = self._node_vars[path_indices_tensor]
        
        return path_means, path_vars

    def get_prediction_index_info(self):
        """
        Get diagnostic information about the prediction index.
        Returns dict with index statistics.
        """
        info = {
            "index_valid": self._prediction_index_valid,
            "total_nodes": len(self._node_to_index) if self._prediction_index_valid else 0,
            "leaf_paths_cached": len(self._leaf_to_path_indices) if self._prediction_index_valid else 0,
            "means_cached": self._node_means is not None,
            "vars_cached": self._node_vars is not None,
        }
        
        if self._prediction_index_valid and self._node_means is not None:
            info["means_shape"] = tuple(self._node_means.shape)
            info["vars_shape"] = tuple(self._node_vars.shape)
            info["device"] = str(self._node_means.device)
        
        return info

    def set_level_weights(self, weights):
        """
        Set custom weights for different tree levels during prediction.
        
        Args:
            weights (list): List of weights for each level [root, level1, level2, ...]
                          Example: [1.0, 2.0, 4.0, 1.0] gives different weights to each level
        """
        self._level_weights = weights
        self._weight_schedule = None  # Clear any schedule when setting manual weights
        # Invalidate prediction index to force rebuild with new weights
        self._invalidate_prediction_index()
        
    def set_weight_schedule(self, schedule_type, max_depth=10, **kwargs):
        """
        Set a weight schedule for different tree levels during prediction.
        
        Args:
            schedule_type (str): Type of schedule - 'constant', 'linear', 'quadratic', 'exponential'
            max_depth (int): Maximum depth to generate weights for
            **kwargs: Additional parameters for specific schedules
                - For 'linear': 'start' (default 1.0), 'end' (default 1.0), 'direction' ('increase'/'decrease')
                - For 'quadratic': 'start_n' (default 1), 'scale' (default 1.0)
                - For 'exponential': 'base' (default 0.5), 'scale' (default 1.0)
        """
        if self._prediction_index_valid:
            max_depth = self.max_depth
        self._weight_schedule = schedule_type
        self._schedule_params = kwargs
        self._level_weights = self._generate_weight_schedule(schedule_type, max_depth, **kwargs)
        # Invalidate prediction index to force rebuild with new weights
        self._invalidate_prediction_index()
        
    def _generate_weight_schedule(self, schedule_type, max_depth, **kwargs):
        """Generate weights based on the specified schedule type."""
        weights = []
        
        if schedule_type == 'constant':
            value = kwargs.get('value', 1.0)
            weights = [value] * max_depth
            
        elif schedule_type == 'linear':
            start = kwargs.get('start', 1.0)
            end = kwargs.get('end', 1.0)
            direction = kwargs.get('direction', 'increase')
            
            if direction == 'decrease':
                start, end = end, start
                
            if max_depth == 1:
                weights = [start]
            else:
                step = (end - start) / (max_depth - 1)
                weights = [start + i * step for i in range(max_depth)]
                
        elif schedule_type == 'quadratic':
            start_n = kwargs.get('start_n', 1)
            
            for i in range(max_depth):
                n = start_n + i
                if n == 0:
                    n = 1  # Skip over 0 to avoid division by zero
                weights.append(1 / (n ** 2))
                
        elif schedule_type == 'exponential':
            base = kwargs.get('base', 0.5)  # Exponential decay base
            
            for i in range(max_depth):
                weights.append((base ** i))

        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        return weights
        
    def get_level_weights(self):
        """Get current level weights"""
        return getattr(self, '_level_weights', [1.0, 1.0, 1.0, 1.0])
        
    def get_weight_schedule_info(self):
        """Get information about the current weight schedule"""
        return {
            'schedule_type': getattr(self, '_weight_schedule', None),
            'schedule_params': getattr(self, '_schedule_params', {}),
            'current_weights': self.get_level_weights()
        }

    def force_rebuild_index(self):
        """Force rebuild of the prediction index"""
        self._invalidate_prediction_index()
        self.build_prediction_index()


    def cobweb_predict_fast(self, input, k=5, return_ids=False, is_embedding=False):
        """
        Ultra-fast prediction using pre-built indices and cached statistics.
        This is now an alias for the indexed prediction method.
        """
        return self.cobweb_predict_indexed(input, k, return_ids, is_embedding)

    def cobweb_predict(self, input, k=5, return_ids=False, is_embedding=False):
        """
        Predict top-k similar entries from the tree.
        
        Args:
            input (str or tensor): Sentence or embedding vector
            k (int): Number of similar items to return
            return_ids (bool): If True, return sentence IDs. Else, return sentences.
            is_embedding (bool): Set True if `input` is already an embedding vector.
        """
        if is_embedding:
            emb = input
        else:
            emb = self.encode_func([input])[0]

        tensor = torch.tensor(emb, device=self.device)
        leaves = self.tree.categorize(tensor, use_best=True, max_nodes=self.max_init_search, retrieve_k=k)

        results = []
        for leaf in leaves:
            sid_lst = getattr(leaf, 'sentence_id', None)
            random.shuffle(sid_lst)  # Shuffle to break ties randomly
            for sid in sid_lst:
                if sid is None or sid >= len(self.sentences):
                    continue
                results.append(sid if return_ids else self.sentences[sid])
        return results

    def print_tree(self):
        """
        Recursively prints the tree structure.
        """
        def _print_node(node, depth=0):
            indent = "  " * depth
            label = f"Sentence ID: {getattr(node, 'sentence_id', 'N/A')}"
            print(f"{indent}- Node ID {node.id} {label}")
            sid = getattr(node, "sentence_id", None)
            if sid is not None and sid < len(self.sentences):
                sentence = self.sentences[sid]
                if sentence is not None:
                    print(f"{indent}    \"{sentence}\"")
                else:
                    print(f"{indent}    [Embedding only]")
            for child in getattr(node, "children", []):
                _print_node(child, depth + 1)

        print("\nCobweb Sentence Clustering Tree:")
        _print_node(self.tree.root)

    def dump_json(self, save_path=None):
        """
        Serializes the CobwebWrapper into a JSON string.
        Only serializes essential data: tree, sentences, and sentence-to-node mapping.
        """
        wrapper_state = {
            "tree": json.loads(self.tree.dump_json()),
            "sentences": self.sentences,
            "embedding_dim": self.tree.shape[0] if hasattr(self.tree, 'shape') else None
        }
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(wrapper_state, f, indent=2)
        return json.dumps(wrapper_state, indent=2)


    @staticmethod
    def load_json(json_data, encode_func=lambda x: x):
        """
        Loads a CobwebWrapper from a JSON string or dict.

        Args:
            json_data (str or dict): The saved wrapper state.
            encode_func (callable): The encoding function to be used.

        Returns:
            CobwebWrapper instance.
        """
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        # Initialize wrapper with minimal setup
        wrapper = CobwebWrapper.__new__(CobwebWrapper)
        wrapper.device = data.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Load tree
        sample_emb = encode_func(["test"])
        embedding_shape = sample_emb.shape[1:]
        tree = CobwebTorchTree(embedding_shape, wrapper.device) # TODO needs to be dumped with initial stuff!
        print("Loading tree from JSON...")
        tree.load_json(json.dumps(data["tree"]))
        
        wrapper.tree = tree
        wrapper.encode_func = encode_func

        # Restore attributes
        wrapper.sentences = data.get("sentences", [])
        wrapper.max_init_search = data.get("max_init_search", 100000)

        # Initialize prediction index attributes
        wrapper._prediction_index_valid = False
        wrapper._node_to_index = {}
        wrapper._index_to_node = {}
        wrapper._node_means = None
        wrapper._node_vars = None
        wrapper._leaf_to_path_indices = {}

        # Reconstruct sentence_to_node mapping
        sentence_to_node = {}
        def _index_nodes(node, st):
            if hasattr(node, 'sentence_id') and node.sentence_id is not None:
                sentence_to_node[node.sentence_id] = node
            for child in getattr(node, "children", []):
                _index_nodes(child, st)

        _index_nodes(wrapper.tree.root, sentence_to_node)
        wrapper.sentence_to_node = sentence_to_node
        # wrapper.print_tree()

        return wrapper

    def __len__(self):
        """
        Returns the number of sentences in the Cobweb tree.
        """
        return len(self.sentences)

    def _visualize_grandparent_tree(self, tree_root, sentences, output_dir="grandparent_trees", num_leaves=6):

        os.makedirs(output_dir, exist_ok=True)

        def get_sentence_label(sid, max_len=250, wrap=40):
            if sid is not None and sid < len(sentences):
                sentence = sentences[sid]
                if sentence:
                    needs_ellipsis = len(sentence) > max_len
                    truncated = sentence[:max_len].rstrip()
                    if needs_ellipsis:
                        truncated += "..."
                    # Wrap at word boundaries every ~wrap characters
                    words = truncated.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line) + len(word) + 1 > wrap:
                            lines.append(current_line)
                            current_line = word
                        else:
                            current_line += (" " if current_line else "") + word
                    if current_line:
                        lines.append(current_line)
                    return "\n".join(lines)
            return None


        def is_leaf_with_sentence(node):
            sid = getattr(node, "sentence_id", None)
            return get_sentence_label(sid) is not None

        def is_grandparent(node):
            # A grandparent is a node whose children have children (i.e., grandchildren exist)
            return any(
                child and getattr(child, "children", None)
                for child in getattr(node, "children", [])
            )

        def collect_grandparents(node):
            result = []
            if is_grandparent(node):
                # Only include this grandparent if it has leaf descendants with valid sentences
                valid_leaf_count = sum(
                    is_leaf_with_sentence(leaf)
                    for child in getattr(node, "children", [])
                    for leaf in getattr(child, "children", [])
                )
                if valid_leaf_count > 0:
                    result.append(node)
            for child in getattr(node, "children", []):
                result.extend(collect_grandparents(child))
            return result

        def get_filename_for_grandparent(node, index=0):
            sid = getattr(node, "sentence_id", None)
            if sid is not None and sid < len(sentences):
                sentence = sentences[sid]
                if sentence:
                    short_hash = hashlib.sha1(sentence.encode()).hexdigest()[:8]
                    return f"gp_{sid}_{short_hash}_{index}.png"
            return f"gp_node_{getattr(node, 'id', 'unknown')}_{index}.png"

        def process_subtree(grandparent_node):
            all_leaves = []
            parent_map = {}

            # First collect only parents/leaves with valid sentences
            for parent in getattr(grandparent_node, "children", []):
                valid_leaves = [leaf for leaf in getattr(parent, "children", []) if is_leaf_with_sentence(leaf)]
                if valid_leaves:
                    parent_map[parent] = valid_leaves
                    all_leaves.extend(valid_leaves)

            if not all_leaves:
                return  # No valid subtree to render

            # Split leaves into batches of 6
            leaf_batches = [all_leaves[i:i + num_leaves] for i in range(0, len(all_leaves), 6)]

            for batch_index, batch in enumerate(leaf_batches):
                dot = Digraph(comment="Grandparent Subtree", format='png')
                dot.attr(rankdir='TB')
                dot.attr('edge', color='lightblue')

                node_ids = {}
                local_counter = {"id": 0}

                def local_next_id():
                    local_counter["id"] += 1
                    return f"n{local_counter['id']}"

                # Grandparent node
                gp_node_id = local_next_id()
                node_ids[grandparent_node] = gp_node_id
                dot.node(gp_node_id, "", shape='circle', width='0.5', style='filled', color='lightblue')

                # Include only relevant parents and children
                for parent, leaves in parent_map.items():
                    # Only include this parent if it has leaves in current batch
                    filtered_leaves = [leaf for leaf in leaves if leaf in batch]
                    if not filtered_leaves:
                        continue

                    parent_id = local_next_id()
                    node_ids[parent] = parent_id
                    dot.node(parent_id, "", shape='circle', width='0.25', style='filled', color='#666666')
                    dot.edge(gp_node_id, parent_id)

                    for leaf in filtered_leaves:
                        sid = getattr(leaf, "sentence_id", None)
                        label = get_sentence_label(sid)
                        if not label:
                            continue  # already filtered, but double-check

                        leaf_id = local_next_id()
                        dot.node(leaf_id, label, shape='box', style='filled', color='lightgrey')
                        dot.edge(parent_id, leaf_id)

                filename = get_filename_for_grandparent(grandparent_node, batch_index)
                filepath = os.path.join(output_dir, filename)
                dot.render(filepath, cleanup=True)
                print(f"Saved: {filepath}")

        grandparents = collect_grandparents(tree_root)
        for gp in grandparents:
            process_subtree(gp)



    def visualize_subtrees(self, directory, num_leaves=6):
        self._visualize_grandparent_tree(self.tree.root, self.sentences, directory, num_leaves)
