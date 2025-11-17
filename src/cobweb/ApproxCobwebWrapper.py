import torch
import json
import random
import math
from tqdm import tqdm
from collections import deque
import numpy as np
import os
import hashlib
from graphviz import Digraph
from src.cobweb.CobwebTorchTree import CobwebTorchTree
from src.cobweb.FastCobwebTorchTree import FastCobwebTorchTree

class ApproxCobwebWrapper:
    def __init__(self, first_method:str, second_method:str, transition_depth:int=1, corpus=None, corpus_embeddings=None, encode_func=lambda x: x):
        """
        Initializes the ApproxCobwebWrapper with optional sentences and/or embeddings.

        Important new parameters:
        *   first_method - can be 'bfs' or 'dfs'
        *   second_method - can be 'pathsum' or 'dot'
        *   transition_depth - the depth at which to collect transition nodes

        The goal is that depending on the first or second method, we compute first method
        to organize a depth of nodes by best semantic similarity and then compute second method
        to find the best semantic similarity from just the first node.

        We need to build a specific type of index for the root, as well as the nodes that describe
        our "transition level" (k-depth or basic-level nodes; for now we use k-depth nodes). For
        the root node, we build a search index or define the method to rank the transition level
        nodes, while for each transition level node, we build an index or define the method to
        rank the leaves.
        *   'bfs' - we don't build an index, and run the Cobweb 'categorize' function without
            the greedy argument from that node
        *   'dfs' - we don't build an index, and run the Cobweb 'categorize' function WITH
            the greedy argument from that node
        *   'pathsum' - we build an index similar to how CobwebWrapper's predict-fast method works
        *   'dot' - we build an index to do kNN with the embeddings under the given node directly
            (defined by node.mean for those nodes)
        """

        self.encode_func = encode_func

        self.first_method = first_method
        self.second_method = second_method
        self.transition_depth = transition_depth

        self.sentences = []
        self.sentence_to_node = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_init_search = 100000

        # Prediction index caching
        self._prediction_index_valid = False
        self._index_to_node = {}
        self._node_to_index = {}
        self._node_means = None
        self._node_vars = None
        self._leaf_to_path_indices = None
        self.max_depth = 0

        # Determine embedding shape
        if corpus_embeddings is not None:
            corpus_embeddings = torch.tensor(corpus_embeddings) if isinstance(corpus_embeddings, list) else corpus_embeddings
            self.embedding_shape = corpus_embeddings.shape[1:]
        elif corpus and len(corpus) > 0:
            sample_emb = self.encode_func([corpus[0]])
            self.embedding_shape = sample_emb.shape[1:]


        self.tree = FastCobwebTorchTree(shape=self.embedding_shape, device=self.device)

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
            leaf = self.tree.fast_ifit(torch.tensor(emb, device=self.device))
            if leaf.sentence_id is None:
                leaf.sentence_id = []
            leaf.sentence_id.append(start_index + i)
            self.sentence_to_node[start_index + i] = leaf

        # Invalidate and rebuild prediction index when new sentences are added
        # so that query-agnostic lookup structures are ready for fast prediction.
        self._invalidate_prediction_index()

    def _invalidate_prediction_index(self):
        """Invalidate the prediction index when tree structure changes"""
        self._prediction_index_valid = False
        self._index_to_node.clear()
        self._node_to_index.clear()
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
        self._node_to_index.clear()
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
            self._node_to_index[hash(node)] = idx
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

        self.tree.analyze_structure()

        print("Building Transition-Node-To-Leaf Datastructure")

        # collecting transition_nodes
        self.transition_nodes = self.tree.categorize_transitions(
            torch.ones(self.embedding_shape, device=self.device),
            transition_depth=self.transition_depth
        )

        print("Number of transition nodes:", len(self.transition_nodes))

        self.t_hash_to_leaf_idxs = {}
        # finding all idxs for leaf

        def get_leaf_idxs(node):
            """Helper method to get leaf idxs."""

            dq = [node]

            res = []

            while len(dq) > 0:
                curr = dq.pop()

                if hasattr(curr, "sentence_id") and len(curr.sentence_id) > 0:
                    res.append(self._node_to_index[hash(curr)])

                for child in curr.children:
                    dq.append(child)
            
            return res

        for tnode in self.transition_nodes:
            self.t_hash_to_leaf_idxs[hash(tnode)] = get_leaf_idxs(tnode)

    def cobweb_predict(self, input, k=5, return_ids=False, is_embedding=False):
        """
        Main two-stage prediction method using configured first_method and
        second_method. This replaces older single-stage prediction helpers and
        centralizes the logic to (1) rank transition nodes by first_method and
        (2) rank leaves under each transition node by second_method.
        """
        # Ensure prediction index is built
        if not self._prediction_index_valid:
            self.build_prediction_index()
        
        if is_embedding:
            emb = input
        else:
            emb = self.encode_func([input])[0]

        # Only allow traversal-based first methods and dense second methods
        if self.first_method not in ('bfs', 'dfs'):
            raise ValueError("first_method must be 'bfs' or 'dfs'")
        if self.second_method not in ('pathsum', 'dot'):
            raise ValueError("second_method must be 'pathsum' or 'dot'")

        x = torch.tensor(emb, device=self.device)  # (D,)

        if len(self._leaf_to_path_indices) == 0:
            return torch.empty(0, device=self.device)
        
        def pathsum_sliced(idxs: list):
            """
            Compute leaf path scores for a subset of indices.
            idxs: list of node indices to include.
            """

            # Slice relevant tensors
            node_means = self._node_means[idxs]           # (num_selected_nodes, dim)
            node_vars = self._node_vars[idxs]             # (num_selected_nodes, dim)
            coo = self._path_matrix.coalesce()
            rows, cols = coo.indices()
            vals = coo.values()

            idxs_tensor = torch.tensor(idxs, device=cols.device, dtype=cols.dtype)

            # Only keep entries whose column index is in idxs
            mask = torch.isin(cols, idxs_tensor)
            rows = rows[mask]
            cols = cols[mask]
            vals = vals[mask]

            # Rebuild the sparse tensor
            path_matrix = torch.sparse_coo_tensor(
                torch.stack([rows, cols]),
                vals,
                size=(self._path_matrix.size(0), len(idxs))
            )

            # Gaussian log-probs (for selected nodes)
            diff_sq = (x.unsqueeze(0) - node_means) ** 2
            node_log_probs = -0.5 * (
                torch.log(node_vars).sum(dim=1)
                + (diff_sq / node_vars).sum(dim=1)
            )  # (num_selected_nodes,)

            # Aggregate along leaf paths
            leaf_scores = torch.sparse.mm(
                path_matrix, node_log_probs.unsqueeze(1)
            ).squeeze(1)  # (num_leaves,)

            return leaf_scores
        
        def dotp_sliced(idxs: list):
            # Slice node means for selected indices
            node_means = self._node_means[idxs]  # (num_selected_nodes, dim)
            # Handle batch or single input
            if x.ndim == 1:
                # Single vector input
                scores = torch.matmul(node_means, x)  # (num_selected_nodes,)
            else:
                # Batched input
                scores = torch.matmul(x, node_means.T)  # (batch, num_selected_nodes)

            return scores

        ranked_tnodes = self.tree.categorize_transitions(
            x,
            transition_depth=self.transition_depth,
            greedy=(self.first_method == 'dfs')
        )

        retrieved = []

        for tnode in ranked_tnodes:

            if len(retrieved) == k:
                break

            if self.second_method == "pathsum":
                scores = pathsum_sliced(self.t_hash_to_leaf_idxs[hash(tnode)])
            else: # dot
                scores = dotp_sliced(self.t_hash_to_leaf_idxs[hash(tnode)])

            topk_idxs = torch.flip(np.argsort(scores.cpu())[-k:], dims=[0])
            retrieved.extend(
                [self._index_to_node[self.t_hash_to_leaf_idxs[hash(tnode)][i]].sentence_id[0]
                    for i in topk_idxs][:min(k - len(retrieved), len(self.t_hash_to_leaf_idxs[hash(tnode)]))]
            )

        if return_ids:
            return retrieved
        else:
            return [self.sentences[i] for i in retrieved]
        
    def exact_cobweb_predict(self, input, k=5, return_ids=False, is_embedding=False):
        """
        Old Cobweb Prediction Function set manually to operate over DFS
        (most similar to approximate retrieval).
        """

        # Ensure prediction index is built
        if not self._prediction_index_valid:
            self.build_prediction_index()
        
        if is_embedding:
            emb = input
        else:
            emb = self.encode_func([input])[0]

        x = torch.tensor(emb, device=self.device)  # (D,)

        res = self.tree.fast_categorize(
            x,
            leaf=True,
            k=k
        )

        if return_ids:
            return [i.sentence_id[0] for i in res]
        else:
            return [self.sentences[i.sentence_id[0]] for i in res]

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

    def print_tree(self):
        """
        Recursively prints the tree structure.
        """
        def _print_node(node, depth=0):
            indent = "  " * depth
            label = f"Sentence ID: {getattr(node, 'sentence_id', 'N/A')}"
            print(f"{indent}- Node ID {node.id} {label}")
            sid = getattr(node, "sentence_id", None)
            if sid and sid[0] < len(self.sentences):
                sentence = self.sentences[sid[0]]
                if sentence is not None:
                    print(f"{indent}    \"{sentence}\"")
                else:
                    print(f"{indent}    [Embedding only]")
            for child in getattr(node, "children", []):
                _print_node(child, depth + 1)

        print("\nCobweb Sentence Clustering Tree:")
        _print_node(self.tree.root)

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
                    # Make intermediary (parent) nodes slightly larger and remove text
                    dot.node(parent_id, "", shape='circle', width='0.35', height='0.35', fixedsize='true', style='filled', color='#666666')
                    dot.edge(gp_node_id, parent_id)

                    for leaf in filtered_leaves:
                        sid = getattr(leaf, "sentence_id", None)
                        label = get_sentence_label(sid)
                        if not label:
                            continue  # already filtered, but double-check

                        leaf_id = local_next_id()
                        # Make leaf nodes' text dominate: larger font, more margin
                        dot.node(leaf_id, label, shape='box', style='filled', color='lightgrey', fontsize='16', fontname='Helvetica', margin='0.2,0.1')
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

    def visualize_query_subtrees(self, query_embeddings, query_texts=None, directory="query_subtrees", k=6, max_nodes_display=500):
        """
        For each query embedding, find top-`k` leaf nodes using `fast_categorize`,
        compute the minimal subtree that contains all those leaves (union of
        ancestor paths), and render that subtree to `directory` (one file per query).

        Args:
            query_embeddings: iterable of embeddings (list, numpy array, or torch tensor).
            directory: output directory for rendered images.
            k: number of top leaves to retrieve per query (passed to `fast_categorize`).
            max_nodes_display: safety cap on number of nodes to render for a single query.
        """

        os.makedirs(directory, exist_ok=True)

        # Allow passing raw query texts instead of precomputed embeddings.
        # If `query_embeddings` is a list of strings, treat them as texts and encode.
        # Otherwise, prefer explicit `query_texts` if provided for labeling.
        q_texts = None
        if isinstance(query_embeddings, list) and len(query_embeddings) > 0 and isinstance(query_embeddings[0], str):
            q_texts = query_embeddings
            q_embs = self.encode_func(q_texts)
            if not torch.is_tensor(q_embs):
                q_embs = torch.tensor(q_embs)
            q_embs = q_embs.to(self.device)
        else:
            # If explicit query_texts provided, keep for labels
            if query_texts is not None:
                q_texts = query_texts

            # Normalize numeric embeddings to torch tensor on device
            if torch.is_tensor(query_embeddings):
                q_embs = query_embeddings.to(self.device)
            else:
                try:
                    q_embs = torch.tensor(query_embeddings, device=self.device)
                except Exception:
                    # Fallback: try encoding if embeddings can't be tensorized
                    if query_texts is not None:
                        q_embs = self.encode_func(query_texts)
                        if not torch.is_tensor(q_embs):
                            q_embs = torch.tensor(q_embs)
                        q_embs = q_embs.to(self.device)
                    else:
                        raise

        def get_sentence_label(sid, max_len=250, wrap=40):
            if sid is not None and sid < len(self.sentences):
                sentence = self.sentences[sid]
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

        # Iterate queries
        for qi in range(q_embs.shape[0] if q_embs.ndim > 1 else 1):
            if q_embs.ndim > 1:
                x = q_embs[qi]
            else:
                x = q_embs

            # get top-k leaf nodes via tree categorize
            try:
                top_nodes = self.tree.fast_categorize(x, leaf=True, k=k)
            except Exception as e:
                print(f"[Warning] fast_categorize failed for query {qi}: {e}")
                continue

            if not top_nodes:
                print(f"Query {qi}: no leaves retrieved")
                continue

            # Collect the actual leaf node objects
            leaf_nodes = list(top_nodes)

            # For each leaf, collect path of ancestors up to root
            nodes_in_subtree = set()
            parent_map = {}

            for leaf in leaf_nodes:
                curr = leaf
                prev = None
                while curr is not None:
                    nodes_in_subtree.add(curr)
                    # remember parent->child mapping for edges
                    par = getattr(curr, 'parent', None)
                    if par is not None:
                        if par not in parent_map:
                            parent_map[par] = set()
                        parent_map[par].add(curr)
                    prev = curr
                    curr = getattr(curr, 'parent', None)

            if len(nodes_in_subtree) == 0:
                print(f"Query {qi}: empty subtree")
                continue

            if len(nodes_in_subtree) > max_nodes_display:
                print(f"Query {qi}: subtree too large ({len(nodes_in_subtree)} nodes), skipping render")
                continue

            # Render subtree with graphviz
            dot = Digraph(comment=f"Query_{qi}_Subtree", format='png')
            dot.attr(rankdir='TB')
            dot.attr('edge', color='lightblue')

            # Create a small boxed node for the query text in the top corner.
            label_text = None
            if q_texts is not None and qi < len(q_texts):
                label_text = q_texts[qi]
            if label_text is None:
                label_text = f"<embedding_{qi}>"

            # Truncate and wrap label to reasonable length and line width
            def wrap_query_text(text, max_len=200, wrap=40):
                if text is None:
                    return "Query:"
                needs_ellipsis = len(text) > max_len
                truncated = text[:max_len].rstrip()
                if needs_ellipsis:
                    truncated += "..."
                words = truncated.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + (1 if current_line else 0) > wrap:
                        lines.append(current_line)
                        current_line = word
                    else:
                        current_line += (" " if current_line else "") + word
                if current_line:
                    lines.append(current_line)
                if not lines:
                    return "Query:"
                # Put 'Query:' on its own first line so the message wraps below it
                return "Query:\n" + "\n".join(lines)

            wrapped_label = wrap_query_text(label_text, max_len=200, wrap=40)

            # Place the query label inside a tiny top-ranked subgraph so it sits at the top.
            qnode_name = f"q{qi}_label"
            with dot.subgraph(name=f"cluster_q_{qi}") as c:
                c.attr(rank='min')
                c.node(qnode_name, wrapped_label, shape='box', fontsize='12', fontname='Helvetica', style='filled,rounded', fillcolor='lightyellow', margin='0.08,0.05')

            node_ids = {}
            local_counter = {"id": 0}

            def local_next_id():
                local_counter["id"] += 1
                return f"n{local_counter['id']}"

            # Create nodes
            for node in nodes_in_subtree:
                nid = local_next_id()
                node_ids[node] = nid
                sid = getattr(node, 'sentence_id', None)
                # If node is a leaf with sentence, label it; otherwise small circle
                if sid and isinstance(sid, list) and len(sid) and sid[0] is not None and sid[0] < len(self.sentences):
                    label = get_sentence_label(sid[0])
                    if not label:
                        label = "[Embedding only]"
                    # Leaf node: emphasize text, larger font and margin
                    dot.node(nid, label, shape='box', style='filled', color='lightgrey', fontsize='16', fontname='Helvetica', margin='0.2,0.1')
                else:
                    # internal node: remove text and make it slightly larger so leaves dominate visually
                    dot.node(nid, "", shape='circle', width='0.35', height='0.35', fixedsize='true', style='filled', color='#ccccff')

            # Create edges only where both parent and child are in subtree
            for parent, children in parent_map.items():
                if parent not in node_ids:
                    continue
                for child in children:
                    if child not in node_ids:
                        continue
                    dot.edge(node_ids[parent], node_ids[child])

            filename = os.path.join(directory, f"query_{qi}_subtree.png")
            filepath = os.path.join(directory, f"query_{qi}_subtree")
            dot.render(filepath, cleanup=True)
            print(f"Saved: {filename}")
