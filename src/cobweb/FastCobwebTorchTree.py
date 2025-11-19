##-------------------------------------------------------------
## FastCobwebTorchTree.py
## Extension to CobwebTorchTree with Python.
##-------------------------------------------------------------

import json
import math
from random import shuffle, random
from math import log, isclose
from collections import defaultdict
import heapq

import torch
from src.cobweb.CobwebTorchNode import CobwebTorchNode # Already has a pointer to the parent!

class FastCobwebTorchTree(object):
    """
    The CobwebTree contains the knowledge base of a particular instance of the
    cobweb algorithm and can be used to fit and categorize instances.
    """

    def __init__(self, shape, use_info=True, acuity_cutoff=False,
                 use_kl=True, prior_var=None, alpha=1e-8, device=None, gradient_flow=False, precompute=10000):
        """
        The tree constructor.
        """
        self.gradient_flow = gradient_flow
        self.device = device
        self.use_info = use_info
        self.acuity_cutoff = acuity_cutoff
        self.use_kl = use_kl
        self.shape = shape
        self.alpha = torch.tensor(alpha, dtype=torch.float, device=self.device,
                                  requires_grad=self.gradient_flow)
        self.pi_tensor = torch.tensor(math.pi, dtype=torch.float,
                                      device=self.device, requires_grad=self.gradient_flow)

        self.prior_var = prior_var
        if prior_var is None:
            self.prior_var = 1 / (2 * math.e * self.pi_tensor)
        # Temperature for soft-DFS weighting (smaller -> closer to argmax)
        self.clear(precompute)

    def clear(self, precompute):
        """
        Clears the concepts of the tree and resets the node map.
        """
        self.root = CobwebTorchNode(shape=self.shape, device=self.device, )
        self.root.tree = self
        # Build node_map for id->node lookup
        self.node_map = {self.root.id: self.root}

        # New variables! Taken directly from the initial CobwebWrapper in the event that categorization-fitting works
        self.idx_to_node = []
        self.hash_to_idx = {}
        self._node_means = torch.zeros(precompute, self.shape[0], 
                                     device=self.device, dtype=torch.float)
        prior_val = (self.prior_var.item() if isinstance(self.prior_var, torch.Tensor)
                     else float(self.prior_var))
        self._node_vars = torch.full((precompute, self.shape[0]), prior_val,
                                     device=self.device, dtype=torch.float)
        # Per-node counts and meanSq stored as tensors so we can vectorize
        self._node_counts = torch.zeros(precompute, device=self.device, dtype=torch.float)
        self._node_meanSq = torch.zeros(precompute, self.shape[0], device=self.device, dtype=torch.float)

        # Coefficients for approximate vectorized pu_for_insert computation.
        # We keep per-edge (child-indexed) coefficients and per-parent aggregated
        # sums so we can compute pu_for_insert(child, instance) in O(D) per node
        # without Python loops over the whole tree.
        D = self.shape[0]
        self._edge_q_base = torch.zeros(precompute, D, device=self.device, dtype=torch.float)
        self._edge_r_base = torch.zeros(precompute, D, device=self.device, dtype=torch.float)
        self._edge_c_base = torch.zeros(precompute, device=self.device, dtype=torch.float)

        self._edge_q_insert = torch.zeros(precompute, D, device=self.device, dtype=torch.float)
        self._edge_r_insert = torch.zeros(precompute, D, device=self.device, dtype=torch.float)
        self._edge_c_insert = torch.zeros(precompute, device=self.device, dtype=torch.float)

        # Aggregated parent sums (over their children) of c_j * base_coeffs
        self._parent_base_q_sum = torch.zeros(precompute, D, device=self.device, dtype=torch.float)
        self._parent_base_r_sum = torch.zeros(precompute, D, device=self.device, dtype=torch.float)
        self._parent_base_c_sum = torch.zeros(precompute, device=self.device, dtype=torch.float)

        # parent index per node (or -1 for root), and number of children per parent
        self._parent_idx = torch.full((precompute,), -1, device=self.device, dtype=torch.long)
        self._children_count = torch.zeros(precompute, device=self.device, dtype=torch.long)
        self._node_to_path_indices = {} # designed to speed up path matrix computations
        self._path_matrix = torch.zeros(precompute, precompute, device=self.device, dtype=torch.float) # precompute this to be large
        self._leaf_idxs = torch.zeros(precompute, device=self.device, dtype=torch.float)

    def resize_structs(self, new_size):
        """
        Resize all preallocated tensors created in clear(), preserving existing data.
        Expands to new_size rows (and columns where appropriate).
        """

        def resize_tensor_1d(t, new_len):
            old_len = t.shape[0]
            result = torch.zeros(new_len, dtype=t.dtype, device=t.device)
            copy_len = min(old_len, new_len)
            result[:copy_len] = t[:copy_len]
            return result

        def resize_tensor_2d(t, new_shape):
            old_rows, old_cols = t.shape
            new_rows, new_cols = new_shape

            result = torch.zeros(new_rows, new_cols, dtype=t.dtype, device=t.device)
            result[:min(old_rows, new_rows), :min(old_cols, new_cols)] = \
                t[:min(old_rows, new_rows), :min(old_cols, new_cols)]
            return result

        D = self.shape[0]

        # ===========================
        # 1. Per-node statistics
        # ===========================
        self._node_means   = resize_tensor_2d(self._node_means,   (new_size, D))
        self._node_vars    = resize_tensor_2d(self._node_vars,    (new_size, D))
        self._node_counts  = resize_tensor_1d(self._node_counts,  new_size)
        self._node_meanSq  = resize_tensor_2d(self._node_meanSq,  (new_size, D))

        # ===========================
        # 2. Edge coefficient tensors
        # ===========================
        self._edge_q_base  = resize_tensor_2d(self._edge_q_base,  (new_size, D))
        self._edge_r_base  = resize_tensor_2d(self._edge_r_base,  (new_size, D))
        self._edge_c_base  = resize_tensor_1d(self._edge_c_base,  new_size)

        self._edge_q_insert = resize_tensor_2d(self._edge_q_insert, (new_size, D))
        self._edge_r_insert = resize_tensor_2d(self._edge_r_insert, (new_size, D))
        self._edge_c_insert = resize_tensor_1d(self._edge_c_insert, new_size)

        # ===========================
        # 3. Aggregated parent sums
        # ===========================
        self._parent_base_q_sum = resize_tensor_2d(self._parent_base_q_sum, (new_size, D))
        self._parent_base_r_sum = resize_tensor_2d(self._parent_base_r_sum, (new_size, D))
        self._parent_base_c_sum = resize_tensor_1d(self._parent_base_c_sum, new_size)

        # ===========================
        # 4. Tree structure metadata
        # ===========================
        self._parent_idx     = resize_tensor_1d(self._parent_idx, new_size).long()
        self._children_count = resize_tensor_1d(self._children_count, new_size).long()

        # Leaf indicators
        self._leaf_idxs = resize_tensor_1d(self._leaf_idxs, new_size)

        # ===========================
        # 5. Path-related matrices
        # ===========================
        self._path_matrix = resize_tensor_2d(self._path_matrix, (new_size, new_size))

    def _build_node_map(self):
        """
        Rebuilds the node_map by traversing the tree.
        """
        self.node_map = {}
        def recurse(node):
            self.node_map[node.id] = node
            for c in node.children:
                recurse(c)
        recurse(self.root)

    def _rebuild_parent_row(self, parent_idx):
        """Rebuild the binary parent->child row for `parent_idx` from live tree."""
        num = len(self.idx_to_node)
        if parent_idx < 0 or parent_idx >= num:
            return
        # reset row
        self._parent_child_binary[parent_idx].zero_()
        parent_node = self.idx_to_node[parent_idx]
        for child in parent_node.children:
            ci = self.hash_to_idx.get(hash(child), None)
            if ci is None:
                continue
            self._parent_child_binary[parent_idx, int(ci)] = 1.0

    def _set_path_row(self, idx):
        """Set both `_path_matrix` (normalized) and `_path_binary` (raw ones) for row `idx`.

        `self._node_to_path_indices[idx]` must exist and contain node indices.
        """
        num_cols = self._path_matrix.size(1)
        row_bin = torch.zeros(num_cols, dtype=self._path_matrix.dtype, device=self.device)
        indices = self._node_to_path_indices.get(idx, [idx])
        if len(indices) > 0:
            inds = torch.tensor(indices, dtype=torch.long, device=self.device)
            row_bin[inds] = 1.0
            self._path_binary[idx] = row_bin

            # normalized row (keep existing behaviour)
            row = torch.zeros(num_cols, dtype=self._path_matrix.dtype, device=self.device)
            row[inds] = 1.0 / float(len(indices))
            self._path_matrix[idx] = row
        else:
            self._path_binary[idx].zero_()
            self._path_matrix[idx].zero_()

    def __str__(self):
        return str(self.root)

    def dump_json(self):
        tree_params = {
                'use_info': self.use_info,
                'acuity_cutoff': self.acuity_cutoff,
                'use_kl': self.use_kl,
                'shape': self.shape.tolist() if isinstance(self.shape, torch.Tensor) else self.shape,
                'alpha': self.alpha.item(),
                'prior_var': self.prior_var.item()}

        json_output = json.dumps(tree_params)[:-1]
        json_output += ', "root": '
        json_output += self.root.iterative_output_json()
        json_output += "}"

        return json_output

    def load_json_helper(self, node_data_json):
        node = CobwebTorchNode(self.shape, device=self.device)
        node.count = torch.tensor(node_data_json['count'], dtype=torch.float,
                                  device=self.device, requires_grad=False)
        node.mean = torch.tensor(node_data_json['mean'], dtype=torch.float,
                                 device=self.device, requires_grad=False)
        node.meanSq = torch.tensor(node_data_json['meanSq'], dtype=torch.float,
                                   device=self.device, requires_grad=False)
        node.sentence_id = node_data_json.get('sentence_id', None)
        return node

    def load_json(self, json_string):
        data = json.loads(json_string)

        self.use_info = data['use_info']
        self.acuity_cutoff = data['acuity_cutoff']
        self.use_kl = data['use_kl']
        self.shape = data['shape']
        self.alpha = torch.tensor(data['alpha'], dtype=torch.float,
                                  device=self.device, requires_grad=False)
        self.prior_var = torch.tensor(data['prior_var'], dtype=torch.float,
                                      device=self.device, requires_grad=False)
        self.root = self.load_json_helper(data['root'])
        self.root.tree = self

        queue = [(self.root, c) for c in data['root']['children']]

        while len(queue) > 0:
            parent, curr_data = queue.pop()
            curr = self.load_json_helper(curr_data)
            curr.tree = self
            curr.parent = parent
            parent.children.append(curr)

            for c in curr_data['children']:
                queue.append((curr, c))

        # after full tree built:
        self._build_node_map()

    def ifit(self, instance, merge_split=True):
        """
        Incrementally fit a new instance into the tree and return its resulting
        concept.

        The instance is passed down the cobweb tree and updates each node to
        incorporate the instance. **This process modifies the tree's
        knowledge** for a non-modifying version of labeling use the
        :meth:`CobwebTree.categorize` function.

        :param instance: An instance to be categorized into the tree.
        :type instance:  :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: CobwebNode

        .. seealso:: :meth:`CobwebTree.cobweb`
        """
        if self.gradient_flow:
            return self.cobweb(instance, merge_split)

        with torch.no_grad():
            return self.cobweb(instance, merge_split)
        
    def fast_ifit(self, instance):
        """
        Fit instances repeatedly using a variant of the categorize function rather than
        the traditional four actions. This will not produce an actual Cobweb Tree, but
        merely a "Cobweb-estimate" that we may still be able to use for prediction.

        Uses `_cobweb_categorize_best_node` for sorting. Note that this will eventually
        be replaced by Pathsum.

        Pipeline:
        *   We use our new _cobweb_categorize_best_node to categorize the best node. Depending
            on where this node is within the tree, we must employ a couple different choices.
            *   If this node is a leaf node, we employ the fringe split case - create a common
                parent to the current node and its parent and then have two descending children.
                *   If this is repeatedly evaluated, we may end up with some form of degenerate
                    tree, so hoping that isn't the case
            *   Otherwise, we can build a child off the given node.
        *   After we have a node, we recurse back up the tree doing the following until we hit
            the root:
            *   We go to the parent and increment counts
        *   We update the pathsum calculation datastructures!

        A little more on updating the pathsum calculation datastructures:
        *   We need to save the node's mean and variance!
        *   We need to update the node's 'global index' and be able to find any node given that
            index (second part not yet confirmed)
        *   Finally, we need to update the path-matrix with a new row and new column
        *   There is no restructuring that takes place, so our two cases are the fringe-split case
            and the regular case.
            *   Fringe-Split: We need to create a new parent for the current node and its child
        """

        if len(self.idx_to_node) >= len(self._node_means) - 1:
            self.resize_structs(len(self._node_means) * 2 + 1)

        def increment_up(current_node):
            # traverse up from the node and increment counts
            parentUp = current_node
            affected_parents = set()
            num_nodes = len(self.idx_to_node)
            dtype = self._path_matrix.dtype
            device = self._path_matrix.device

            while parentUp:
                parentUp.increment_counts(instance)

                # update mean/vars in preallocated buffers and collect affected parents
                if hash(parentUp) in self.hash_to_idx:
                    idx_up = self.hash_to_idx[hash(parentUp)]
                    self._node_means[idx_up] = parentUp.mean
                    self._node_vars[idx_up] = parentUp.var
                    # keep counts/meanSq in sync for vectorized PU computations
                    try:
                        self._node_counts[idx_up] = parentUp.count
                    except Exception:
                        # parentUp.count should be a torch scalar, but be defensive
                        self._node_counts[idx_up] = float(parentUp.count)

                    self._node_meanSq[idx_up] = parentUp.meanSq
                    affected_parents.add(int(idx_up))

                if hasattr(parentUp, "parent") and parentUp.parent:
                    parentUp = parentUp.parent
                else:
                    parentUp = None

            # After updating counts up the chain, recompute per-parent PU coefficients
            # only for the affected parents we collected. This updates per-edge
            # coefficients for each child of the affected parents and the parent's
            # aggregated base sums. Doing this here keeps the work local to the
            # modified path and avoids full-tree rebuilds.
            if len(affected_parents) == 0:
                return

            counts = self._node_counts[:num_nodes].to(dtype=dtype)
            means = self._node_means[:num_nodes]
            vars = self._node_vars[:num_nodes].clamp(min=1e-12)

            for p in affected_parents:
                # safety
                if p < 0 or p >= num_nodes:
                    continue

                parent_node = self.idx_to_node[p]
                children = parent_node.children

                # reset parent aggregated sums
                self._parent_base_q_sum[p].zero_()
                self._parent_base_r_sum[p].zero_()
                self._parent_base_c_sum[p] = 0.0

                # update children_count
                self._children_count[p] = len(children)

                for child in children:
                    ci = self.hash_to_idx.get(hash(child), None)
                    if ci is None:
                        continue
                    i = int(ci)

                    # ensure child stats are refreshed
                    self._node_means[i] = child.mean.to(device=device, dtype=dtype)
                    self._node_vars[i] = child.var.to(device=device, dtype=dtype)
                    self._node_meanSq[i] = child.meanSq.to(device=device, dtype=dtype)
                    try:
                        self._node_counts[i] = child.count
                    except Exception:
                        self._node_counts[i] = float(child.count)

                    Cp = counts[p].item()
                    Ci = counts[i].item()

                    mu_p = means[p]
                    mu_i = means[i]
                    var_p = vars[p]
                    var_i = vars[i]

                    beta_p = 1.0 / (Cp + 1.0)
                    alpha_p = (Cp / (Cp + 1.0)) * mu_p

                    beta_i = 1.0 / (Ci + 1.0)
                    alpha_i = (Ci / (Ci + 1.0)) * mu_i

                    mu_diff = mu_i - alpha_p

                    q_base = 0.5 * (beta_p * beta_p) / var_p
                    r_base = 0.5 * (-2.0 * mu_diff * beta_p) / var_p
                    c_base_vec = 0.5 * (var_i / var_p + (mu_diff * mu_diff) / var_p + torch.log(var_p) - torch.log(var_i) - 1.0)
                    c_base = c_base_vec.sum()

                    delta_beta = beta_i - beta_p
                    delta_alpha = alpha_i - alpha_p

                    q_ins = 0.5 * (delta_beta * delta_beta) / var_p
                    r_ins = 0.5 * (2.0 * delta_alpha * delta_beta) / var_p
                    c_ins_vec = 0.5 * (var_i / var_p + (delta_alpha * delta_alpha) / var_p + torch.log(var_p) - torch.log(var_i) - 1.0)
                    c_ins = c_ins_vec.sum()

                    D = means.size(1)
                    self._edge_q_base[i, :D] = q_base
                    self._edge_r_base[i, :D] = r_base
                    self._edge_c_base[i] = c_base

                    self._edge_q_insert[i, :D] = q_ins
                    self._edge_r_insert[i, :D] = r_ins
                    self._edge_c_insert[i] = c_ins

                    # set parent mapping for child index
                    self._parent_idx[i] = p

                    cj = counts[i]
                    self._parent_base_q_sum[p] += cj * q_base
                    self._parent_base_r_sum[p] += cj * r_base
                    self._parent_base_c_sum[p] += cj * c_base


        ### CHOOSING NODE TO CREATE NEW NODE OFF OF
        current = self.cobweb_no_merge_split(instance)

        if not current.children and (current.is_exact_match(instance) or
                                         current.count == 0):

            # append node first so internal index mappings are valid for
            # vectorized/increment_up computations
            idx = len(self.idx_to_node)
            self.idx_to_node.append(current)
            self.hash_to_idx[hash(current)] = idx

            increment_up(current)

            # update pathsum datastructures
            if current.parent:
                self._node_to_path_indices[idx] = self._node_to_path_indices[self.hash_to_idx[hash(current.parent)]] + [idx]
            else:
                self._node_to_path_indices[idx] = [idx]
            # build a full row with ones at the path indices (match path_matrix dtype/device)
            row = torch.zeros(self._path_matrix.size(1), dtype=self._path_matrix.dtype, device=self.device)
            indices = self._node_to_path_indices[idx]
            if len(indices) > 0:
                inds = torch.tensor(indices, dtype=torch.long, device=self.device)
                row[inds] = 1.0 / float(len(indices))
            self._path_matrix[idx] = row

            self._leaf_idxs[self.hash_to_idx[hash(current)]] = 1

            return current

        elif not current.children:
            # fringe split
            # before: parent -> current
            # after: parent -> new -> current and new -> newChild
            new = CobwebTorchNode(shape=self.shape, device=self.device, otherNode=current)
            current.parent = new
            new.children.append(current)

            if new.parent:
                new.parent.children.remove(current)
                new.parent.children.append(new)
            else:
                self.root = new
            
            newChild = new.create_new_child(instance)

            # update pathsum datastructures (for both newChild and new)

            # steps for integrating our stuff (in order):
            # for new - steal current's path and use it (swap current index true with new index true)
            # for current - add new as an index
            # for newChild - steal new's path and use it (just add newChild index true)

            # add 'new' and 'newChild' as nodes
            self.hash_to_idx[hash(new)] = len(self.idx_to_node)
            self.idx_to_node.append(new)

            self.hash_to_idx[hash(newChild)] = len(self.idx_to_node)
            self.idx_to_node.append(newChild)

            increment_up(newChild)

            newIdx = self.hash_to_idx[hash(new)]
            currIdx = self.hash_to_idx[hash(current)]
            newChildIdx = self.hash_to_idx[hash(newChild)]

            # structure changed: per-parent coefficients updated inside increment_up

            # specific changes for node 'new'
            self._node_to_path_indices[newIdx] = self._node_to_path_indices[currIdx] + [newIdx]
            self._node_to_path_indices[newIdx].remove(currIdx)
            # rebuild normalized row for newIdx
            new_inds = self._node_to_path_indices[newIdx]
            new_row = torch.zeros(self._path_matrix.size(1), dtype=self._path_matrix.dtype, device=self.device)
            if len(new_inds) > 0:
                new_row[torch.tensor(new_inds, dtype=torch.long, device=self.device)] = 1.0 / float(len(new_inds))
            self._path_matrix[newIdx] = new_row

            # specific changes for node 'current' (append newIdx)
            self._node_to_path_indices[currIdx].append(newIdx)
            curr_inds = self._node_to_path_indices[currIdx]
            curr_row = torch.zeros(self._path_matrix.size(1), dtype=self._path_matrix.dtype, device=self.device)
            if len(curr_inds) > 0:
                curr_row[torch.tensor(curr_inds, dtype=torch.long, device=self.device)] = 1.0 / float(len(curr_inds))
            self._path_matrix[currIdx] = curr_row

            # specific changes for node 'child'
            self._node_to_path_indices[newChildIdx] = self._node_to_path_indices[newIdx] + [newChildIdx]
            child_inds = self._node_to_path_indices[newChildIdx]
            child_row = torch.zeros(self._path_matrix.size(1), dtype=self._path_matrix.dtype, device=self.device)
            if len(child_inds) > 0:
                child_row[torch.tensor(child_inds, dtype=torch.long, device=self.device)] = 1.0 / float(len(child_inds))
            self._path_matrix[newChildIdx] = child_row

            self._leaf_idxs[newChildIdx] = 1

            return newChild

        else:
            newChild = current.create_new_child(instance)

            # append the new child and set mapping before incrementing up
            idx = len(self.idx_to_node)
            self.idx_to_node.append(newChild)
            self.hash_to_idx[hash(newChild)] = idx

            increment_up(newChild)

            # update pathsum datastructures
            self._node_to_path_indices[idx] = self._node_to_path_indices[self.hash_to_idx[hash(newChild.parent)]] + [idx]
            row = torch.zeros(self._path_matrix.size(1), dtype=self._path_matrix.dtype, device=self.device)
            indices = self._node_to_path_indices[idx]
            if len(indices) > 0:
                inds = torch.tensor(indices, dtype=torch.long, device=self.device)
                row[inds] = 1.0 / float(len(indices))
            self._path_matrix[idx] = row

            # NEED TO STORE ALL LEAF IDXS
            self._leaf_idxs[self.hash_to_idx[hash(newChild)]] = 1
            self._leaf_idxs[self.hash_to_idx[hash(current)]] = 0

            return newChild
    
    def fast_categorize(self, instance, leaf=False, k=1) -> list:
        """
        An adapted function that uses the CobwebWrapper Pathsum definitions to do prediction!

        instance: the embedding to sort down!

        Returns a single node for the node returned.
        """
        
        scores = self.fast_categorize_scores(instance)

        if scores == []:
            return [self.root]

        if leaf:
            scores = scores * self._leaf_idxs[:len(self.idx_to_node)]

        def get_topk_nonzero_indices(tensor_1d, k):
            nonzero_indices = torch.nonzero(tensor_1d, as_tuple=False).squeeze()
            if nonzero_indices.dim() == 0:
                nonzero_indices = nonzero_indices.unsqueeze(0)
            
            num_nonzero = nonzero_indices.numel()
            k = min(k, num_nonzero)
            
            nonzero_values = tensor_1d[nonzero_indices]

            _, indices_of_topk = torch.topk(nonzero_values, k, largest=True, sorted=True)
            
            topk_nonzero_indices = nonzero_indices[indices_of_topk]
            
            return topk_nonzero_indices
        
        indices = get_topk_nonzero_indices(scores, k)

        return [self.idx_to_node[i] for i in indices]

    def fast_categorize_scores(self, instance):
        """
        An adapted function that uses the CobwebWrapper Pathsum definitions to do prediction!

        Returns a vector attributing to the scores of the categorization function.
        """

        if len(self.idx_to_node) == 0:
            return []

        x = instance.to(self.device)

        if len(self._node_to_path_indices) == 0:
            return torch.empty(0, device=self.device)

        # Gaussian log-probs computed only over active nodes
        num_nodes = len(self.idx_to_node)
        if num_nodes == 0:
            return torch.empty(0, device=self.device)

        node_means = self._node_means[:num_nodes]
        node_vars = self._node_vars[:num_nodes]
        # avoid zeros/nans
        eps = 1e-12
        node_vars = node_vars.clamp(min=eps)

        diff_sq = (x.unsqueeze(0) - node_means) ** 2
        node_log_probs = -0.5 * (torch.log(node_vars).sum(dim=1) + (diff_sq / node_vars).sum(dim=1))
        node_log_probs = node_log_probs.to(self._path_matrix.dtype)

        # Multiply path matrix (rows x num_nodes) with node_log_probs
        pm = self._path_matrix[:, :num_nodes]
        node_scores = pm.matmul(node_log_probs.unsqueeze(1)).squeeze(1)

        # return only the first num_nodes entries (consistent with usage)
        return node_scores[:num_nodes]

    def cobweb(self, instance, merge_split):
        """
        The core cobweb algorithm used in fitting and categorization.

        In the general case, the cobweb algorithm entertains a number of
        sorting operations for the instance and then commits to the operation
        that maximizes the :meth:`category utility
        <CobwebNode.category_utility>` of the tree at the current node and then
        recurses.

        At each node the alogrithm first calculates the category utility of
        inserting the instance at each of the node's children, keeping the best
        two (see: :meth:`CobwebNode.two_best_children
        <CobwebNode.two_best_children>`), and then calculates the
        category_utility of performing other operations using the best two
        children (see: :meth:`CobwebNode.get_best_operation
        <CobwebNode.get_best_operation>`), commiting to whichever operation
        results in the highest category utility. In the case of ties an
        operation is chosen at random.

        In the base case, i.e. a leaf node, the algorithm checks to see if
        the current leaf is an exact match to the current node. If it is, then
        the instance is inserted and the leaf is returned. Otherwise, a new
        leaf is created.

        .. note:: This function is equivalent to calling
            :meth:`CobwebTree.ifit` but its better to call ifit because it is
            the polymorphic method siganture between the different cobweb
            family algorithms.

        :param instance: an instance to incorporate into the tree
        :type instance: :ref:`Instance<instance-rep>`
        :return: a concept describing the instance
        :rtype: CobwebNode

        .. seealso:: :meth:`CobwebTree.ifit`, :meth:`CobwebTree.categorize`
        """
        current = self.root

        while current:
            # the current.count == 0 here is for the initially empty tree.
            if not current.children and (current.is_exact_match(instance) or
                                         current.count == 0):
                # print("root match or leaf match")
                current.increment_counts(instance)
                break

            elif not current.children:
                # print("fringe split")
                new = CobwebTorchNode(shape=self.shape, device=self.device, otherNode=current)
                current.parent = new
                new.children.append(current)

                if new.parent:
                    new.parent.children.remove(current)
                    new.parent.children.append(new)
                else:
                    self.root = new

                new.increment_counts(instance)
                current = new.create_new_child(instance)
                break

            else:
                best1_pu, best1, best2 = current.two_best_children(instance)

                _, best_action = current.get_best_operation(instance, best1,
                                                            best2, best1_pu, merge_split)

                if best_action == 'best':
                    current.increment_counts(instance)
                    current = best1
                elif best_action == 'new':
                    current.increment_counts(instance)
                    current = current.create_new_child(instance)
                    break
                elif best_action == 'merge':
                    current.increment_counts(instance)
                    new_child = current.merge(best1, best2)
                    current = new_child
                elif best_action == 'split':
                    current.split(best1)
                else:
                    raise Exception('Best action choice "' + best_action +
                                    '" not a recognized option. This should be'
                                    ' impossible...')
        return current

    def cobweb_no_merge_split(self, instance):
        """
        Cobweb insertion that disables the merge and split operations.

        This method first precomputes, for every node in the tree, the
        utility of creating a new child (`pu_for_new_child`) and the best
        child's utility (`pu_for_insert`) for the given `instance`. After
        the precomputation step it performs the usual cobweb loop but only
        entertains the 'best' and 'new' operations (no merge/split).

        Precomputing avoids repeatedly recomputing the same utilities while
        descending the tree.
        """

        # Compute vectorized scores once (child-level and parent-level)
        num_nodes = len(self.idx_to_node)
        if num_nodes == 0:
            return self.root

        S_child = self.vectorized_pu_insert_scores(instance)
        P_parent = self.vectorized_pu_new_scores(instance)

        # sanitize
        S_child = torch.nan_to_num(S_child, nan=-1e9, neginf=-1e9, posinf=1e9)
        P_parent = torch.nan_to_num(P_parent, nan=1e9, neginf=-1e9, posinf=1e9)

        # Fast tree traversal using precomputed scores. We never mutate the tree.
        current = self.root

        while True:
            # If leaf: return it (caller can decide whether to fringe-split)
            if not current.children:
                return current

            # Parent index for current
            p_idx = self.hash_to_idx.get(hash(current), None)

            # Find best child among existing children using S_child precomputed
            best_child = None
            best_score = float('-inf')

            for ch in current.children:
                ci = self.hash_to_idx.get(hash(ch), None)
                if ci is None or ci >= S_child.size(0):
                    continue
                score = float(S_child[int(ci)].item())
                if best_child is None or score > best_score:
                    best_child = ch
                    best_score = score

            # parent new score
            parent_new_score = float(P_parent[int(p_idx)].item()) if p_idx < P_parent.size(0) else float('inf')

            # If best child yields higher (or equal) score, descend
            if best_child is not None and best_score >= parent_new_score:
                current = best_child
                continue
            # otherwise terminate at this parent
            return current

    def _rebuild_vectorized_coeffs(self):
        """
        Lightweight initializer for vectorized maps used by the scoring
        routines. Heavy per-edge coefficient computation is intended to be
        performed incrementally in `increment_up` during fitting; this
        function only populates index maps, per-node stats and path rows
        when called (used for cold-start / first call).
        """
        # If mapping already exists, do nothing (incremental updates will
        # keep tensors in sync during fitting).
        if len(self.idx_to_node) > 0:
            return

        # Build idx_to_node (preorder traversal) and map hashes to indices
        self.idx_to_node = []
        self.hash_to_idx = {}

        stack = [self.root]
        while stack:
            node = stack.pop()
            idx = len(self.idx_to_node)
            self.idx_to_node.append(node)
            self.hash_to_idx[hash(node)] = idx
            for c in reversed(node.children):
                stack.append(c)

        num_nodes = len(self.idx_to_node)
        if num_nodes == 0:
            return

        dtype = self._path_matrix.dtype
        device = self._path_matrix.device

        # Fill node-level stats (do not compute per-edge coefficients here)
        for i, node in enumerate(self.idx_to_node):
            self._node_means[i] = node.mean.to(device=device, dtype=dtype)
            self._node_vars[i] = node.var.to(device=device, dtype=dtype)
            self._node_meanSq[i] = node.meanSq.to(device=device, dtype=dtype)
            try:
                self._node_counts[i] = node.count
            except Exception:
                self._node_counts[i] = float(node.count)

            # set parent index if available
            if hasattr(node, 'parent') and node.parent and hash(node.parent) in self.hash_to_idx:
                self._parent_idx[i] = self.hash_to_idx[hash(node.parent)]
            else:
                self._parent_idx[i] = -1

        # Build path indices and normalized path matrix rows
        for idx, node in enumerate(self.idx_to_node):
            path = []
            cur = node
            while cur is not None:
                h = hash(cur)
                if h in self.hash_to_idx:
                    path.insert(0, self.hash_to_idx[h])
                cur = cur.parent if hasattr(cur, 'parent') and cur.parent else None
            self._node_to_path_indices[idx] = path if path else [idx]

            row = torch.zeros(self._path_matrix.size(1), dtype=self._path_matrix.dtype, device=device)
            indices = self._node_to_path_indices[idx]
            if len(indices) > 0:
                inds = torch.tensor(indices, dtype=torch.long, device=device)
                row[inds] = 1.0 / float(len(indices))
            self._path_matrix[idx] = row

        # Leaf indices
        self._leaf_idxs[:num_nodes] = 0
        for idx, node in enumerate(self.idx_to_node):
            if len(node.children) == 0:
                self._leaf_idxs[idx] = 1

    def _cobweb_categorize_best_node(self, instance, greedy=False, max_nodes=float('inf')):
        """
        A cobweb specific version of categorize, not intended to be
        externally called.

        TODO: We will eventually replace this with PathSum!

        .. seealso:: :meth:`CobwebTree.categorize`
        """
        queue = []
        heapq.heappush(queue, (-self.root.log_prob(instance), 0.0, random(), self.root))
        nodes_visited = 0

        best = self.root
        best_score = float('-inf')

        while len(queue) > 0:
            if greedy:
                neg_score, neg_curr_ll, _, curr = queue.pop()
            else:
                neg_score, neg_curr_ll, _, curr = heapq.heappop(queue)
            score = -neg_score # the heap sorts smallest to largest, so we flip the sign
            curr_ll = -neg_curr_ll # the heap sorts smallest to largest, so we flip the sign
            nodes_visited += 1

            if score > best_score:
                best = curr
                best_score = score

            if nodes_visited >= max_nodes:
                break

            if len(curr.children) > 0:
                ll_children_unnorm = torch.zeros(len(curr.children))
                # for i, c in enumerate(curr.children):
                #     log_prob = c.log_prob(instance)
                #     ll_children_unnorm[i] = (log_prob + math.log(c.count) - math.log(curr.count))
                # log_p_of_x = torch.logsumexp(ll_children_unnorm, dim=0)

                add = []

                for i, c in enumerate(curr.children):
                    # child_ll = ll_children_unnorm[i] - log_p_of_x + curr_ll
                    child_ll_inst = c.log_prob(instance)
                    child_score =  child_ll_inst #score + child_ll 
                    # child_score = child_ll + child_ll_inst # p(c|x) * p(x|c)
                    if greedy:
                        add.append((-child_score, -score, random(), c))
                    else:
                        heapq.heappush(queue, (-child_score, -score, random(), c))

                if greedy:
                    add.sort()  # sort by neg_score
                    queue.extend(add[::-1]) # reverses so that most optimal element is at the end

        return best

    def _cobweb_categorize(self, instance, use_best, greedy, max_nodes, retrieve_k=None):
        """
        A cobweb specific version of categorize, not intended to be
        externally called.

        .. seealso:: :meth:`CobwebTree.categorize`
        """
        queue = []
        heapq.heappush(queue, (-self.root.log_prob(instance), 0.0, random(), self.root))
        nodes_visited = 0

        best = self.root
        best_score = float('-inf')

        retrieved = []

        while len(queue) > 0:
            if greedy:
                neg_score, neg_curr_ll, _, curr = queue.pop()
            else:
                neg_score, neg_curr_ll, _, curr = heapq.heappop(queue)
            score = -neg_score # the heap sorts smallest to largest, so we flip the sign
            curr_ll = -neg_curr_ll # the heap sorts smallest to largest, so we flip the sign
            nodes_visited += 1

            if score > best_score:
                best = curr
                best_score = score

            if nodes_visited >= max_nodes:
                break

            if curr.sentence_id:
                heapq.heappush(retrieved, (len(retrieved), random(), curr))

            if retrieve_k is not None and len(retrieved) == retrieve_k:
                break # TODO can replace this with a part at the end optionally!

            if len(curr.children) > 0:
                ll_children_unnorm = torch.zeros(len(curr.children))
                # for i, c in enumerate(curr.children):
                #     log_prob = c.log_prob(instance)
                #     ll_children_unnorm[i] = (log_prob + math.log(c.count) - math.log(curr.count))
                # log_p_of_x = torch.logsumexp(ll_children_unnorm, dim=0)

                add = []

                for i, c in enumerate(curr.children):
                    # child_ll = ll_children_unnorm[i] - log_p_of_x + curr_ll
                    child_ll_inst = c.log_prob(instance)
                    child_score =  child_ll_inst #score + child_ll 
                    # child_score = child_ll + child_ll_inst # p(c|x) * p(x|c)
                    if greedy:
                        add.append((-child_score, -score, random(), c))
                    else:
                        heapq.heappush(queue, (-child_score, -score, random(), c))

                if greedy:
                    add.sort()  # sort by neg_score
                    queue.extend(add[::-1]) # reverses so that most optimal element is at the end

        if retrieve_k is None:
            return best if use_best else curr

        return [retrieved[i][-1] for i in range(retrieve_k)]

    def categorize_transitions(self, instance, transition_depth, use_best=True, greedy=False, max_nodes=float('inf')):
        """
        Discover and return nodes at exactly `transition_depth` in traversal order.

        This is similar to the regular categorize search but will NOT
        expand children of nodes once they are at the transition depth. The
        returned nodes are the transition-level nodes ordered by the same
        scoring/priority used by `_cobweb_categorize`.

        Args:
            instance: instance to score against the tree
            transition_depth: integer depth (root==0) at which to collect nodes
            use_best, greedy, max_nodes, retrieve_k: same semantics as categorize

        Returns:
            If retrieve_k is None: returns best node (same as categorize semantics)
            Otherwise: returns a list of up to `retrieve_k` nodes at the transition depth
        """
        queue = []
        # queue entries: (neg_score, neg_curr_ll, rand, node, depth)
        heapq.heappush(queue, (-self.root.log_prob(instance), 0.0, random(), self.root, 0))
        nodes_visited = 0

        best = self.root
        best_score = float('-inf')

        retrieved = []

        while len(queue) > 0:
            if greedy:
                neg_score, neg_curr_ll, _, curr, depth = queue.pop()
            else:
                neg_score, neg_curr_ll, _, curr, depth = heapq.heappop(queue)

            score = -neg_score
            curr_ll = -neg_curr_ll
            nodes_visited += 1

            if score > best_score:
                best = curr
                best_score = score

            if nodes_visited >= max_nodes:
                break

            # If this node is at the transition depth and is a valid concept
            # (has sentence_id or is a parent), collect it but DO NOT expand its children
            if depth == transition_depth:
                retrieved.append(curr)
                continue

            # Otherwise, behave like _cobweb_categorize and consider children
            if len(curr.children) > 0:
                add = []
                for i, c in enumerate(curr.children):
                    child_ll_inst = c.log_prob(instance)
                    child_score = child_ll_inst
                    if greedy:
                        add.append((-child_score, score, random(), c, depth + 1))
                    else:
                        heapq.heappush(queue, (-child_score, score, random(), c, depth + 1))

                if greedy:
                    add.sort()
                    queue.extend(add[::-1])

        return retrieved

    def categorize(self, instance, use_best=True, greedy=False, max_nodes=float('inf'), retrieve_k=None):
        """
        Sort an instance in the categorization tree and return its resulting
        concept.

        The instance is passed down the categorization tree according to the
        normal cobweb algorithm except using only the best operator and without
        modifying nodes' probability tables. **This process does not modify the
        tree's knowledge** for a modifying version of labeling use the
        :meth:`CobwebTree.ifit` function

        :param instance: an instance to be categorized into the tree.
        :type instance: :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: CobwebNode

        .. seealso:: :meth:`CobwebTree.cobweb`
        """
        # If a transition_depth is provided, use the specialized traversal
        # that collects nodes at exactly that depth and does not expand
        # beyond them. This is used by the wrapper to score transition-level
        # nodes without exploring deeper leaves.

        if self.gradient_flow:
            return self._cobweb_categorize(instance, use_best, greedy, max_nodes, retrieve_k)

        with torch.no_grad():
            return self._cobweb_categorize(instance, use_best, greedy, max_nodes, retrieve_k)

    def compute_var(self, meanSq, count):
        # return (meanSq + 30*1) / (count + 30)

        if self.acuity_cutoff:
            return torch.clamp(meanSq / count, self.prior_var) # with cutoff
        else:
            return meanSq / count + self.prior_var # with adjustment

    def compute_score(self, mu1, var1, mu2, var2):
        if (self.use_info):
            if (self.use_kl):
                # score2 = (0.5 * (torch.log(var2) - torch.log(var1)) +
                #          (var1 + torch.pow(mu1 - mu2, 2))/(2 * var2) -
                #          0.5).sum()
                score = (torch.log(var2) - torch.log(var1)).sum()
                score += ((var1 + torch.pow(mu1 - mu2, 2))/(var2)).sum()
                score -= mu1.numel()
                score /= 2

                # if torch.abs(score - score2) > 1e-3:
                #     print(score - score2)

            else:
                score = 0.5 * (torch.log(var2) - torch.log(var1)).sum()
        else:
            score = -(1 / (2 * torch.sqrt(self.pi_tensor) * torch.sqrt(var1))).sum()
            score += (1 / (2 * torch.sqrt(self.pi_tensor) * torch.sqrt(var2))).sum()

        return score

    def soft_find_first_new_greedy(self, instance):
        """
        Implementation notes / heuristics:
        - We compute a per-node log-likelihood of the instance under
            the node's Gaussian (same form used elsewhere in this class),
            then invert that to obtain a "newiness" score (low likelihood
            => high newiness).
        - We aggregate per-node newiness along the stored path matrix
            using a single matrix multiplication, which simulates the
            top-down accumulation used by greedy search.
        - The node with the largest aggregated newiness is returned as
            the approximated 'first new' node. No loops are used.

        This is an approximation and intentionally conservative — it
        returns one of the nodes in `idx_to_node` (or `None` if empty).

        Args:
                instance: tensor instance (same format as used elsewhere)
                temperature: (unused currently) kept for API compatibility

        Returns:
                A node from `self.idx_to_node` or `None` if none available.
        """
        if len(self.idx_to_node) == 0:
            return self.root

        x = instance.to(self.device)

        num_nodes = len(self.idx_to_node)
        node_means = self._node_means[:num_nodes]
        node_vars = self._node_vars[:num_nodes]

        eps = 1e-12
        node_vars = node_vars.clamp(min=eps)

        diff_sq = (x.unsqueeze(0) - node_means) ** 2
        scaled_diff = diff_sq / node_vars
        sq_term = scaled_diff.sum(dim=1)

        log_var_sum = torch.log(node_vars).sum(dim=1)
        node_log_probs = -0.5 * (log_var_sum + sq_term)

        # Invert log-probs to get a "newiness" score: nodes that fit the
        # instance poorly (low log-prob) get higher newiness.
        newiness = -node_log_probs

        new_mean = newiness.mean()
        new_std = newiness.std(unbiased=False)
        new_std = new_std.clamp(min=eps)
        newiness = (newiness - new_mean) / new_std

        pm = self._path_matrix[:num_nodes, :num_nodes]
        agg_scores = pm.matmul(newiness)

        best_idx = int(torch.argmax(agg_scores).item())

        if best_idx < 0 or best_idx >= len(self.idx_to_node):
            return None

        return self.idx_to_node[best_idx]

    def analyze_structure(self):
        """
        Analyze the structure of the tree:
        - Print total number of leaf nodes.
        - Print the number of nodes at each depth level (via BFS).
        - Print a histogram of number of children per parent node.
        """
        from collections import deque, defaultdict

        leaf_count = 0
        level_counts = defaultdict(int)
        child_histogram = defaultdict(int)

        queue = deque([(self.root, 0)])

        while queue:
            node, level = queue.popleft()
            level_counts[level] += 1

            if not node.children:
                leaf_count += 1
            else:
                num_children = len(node.children)
                child_histogram[num_children] += 1
                for child in node.children:
                    queue.append((child, level + 1))

        print(f"\nTotal number of leaf nodes: {leaf_count}\n")

        print("Number of nodes at each level:")
        for level in sorted(level_counts.keys()):
            print(f"  Level {level}: {level_counts[level]} node(s)")

        print("\nParent nodes by number of children:")
        for num_children in sorted(child_histogram.keys()):
            print(f" {child_histogram[num_children]} parent(s) with {num_children} child(ren)")

    def build_node_depths(self):
        """
        Build (or rebuild) a tensor `self._node_depths` containing the depth
        (root=0) for each stored node index. This is a lazy helper and may
        be called occasionally (it is not used in inner loops).
        """
        num_nodes = len(self.idx_to_node)
        if num_nodes == 0:
            self._node_depths = torch.empty(0, dtype=torch.long, device=self.device)
            return self._node_depths

        depths = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
        for idx in range(num_nodes):
            path = self._node_to_path_indices.get(idx, [idx])
            depths[idx] = max(0, len(path) - 1)

        self._node_depths = depths
        return self._node_depths

    def vectorized_pu_insert_scores(self, instance):
        """
        Approximate and compute `parent.pu_for_insert(child, instance)` for
        all stored nodes `child` using precomputed coefficients. Returns a
        1-D tensor of length `num_nodes` with the score for each node. This
        uses the quadratic approximation derived in `rebuild_pu_coeffs`.
        """
        num_nodes = len(self.idx_to_node)
        if num_nodes == 0:
            return torch.empty(0, device=self.device)

        dtype = self._path_matrix.dtype
        device = self._path_matrix.device

        x = instance.to(device=device, dtype=dtype)
        x_sq = x * x

        # parent-level aggregated base sums
        parent_q = self._parent_base_q_sum[:num_nodes]
        parent_r = self._parent_base_r_sum[:num_nodes]
        parent_c = self._parent_base_c_sum[:num_nodes]

        parent_idx = self._parent_idx[:num_nodes]
        children_count = self._children_count[:num_nodes].to(dtype=dtype)
        counts = self._node_counts[:num_nodes].to(dtype=dtype)

        # Vectorized computation for all nodes at once.
        # Prepare per-child views of parent aggregates (safe for root indices).
        num = num_nodes

        # Avoid indexing with -1: create a safe parent index where roots point to 0
        safe_parent_idx = parent_idx.clone()
        if safe_parent_idx.numel() > 0:
            safe_parent_idx = safe_parent_idx.to(dtype=torch.long, device=device)
            root_mask = safe_parent_idx < 0
            if root_mask.any():
                safe_parent_idx[root_mask] = 0
        else:
            root_mask = torch.zeros(0, dtype=torch.bool, device=device)

        # parent aggregated sums per child (num_nodes x D)
        parent_q_per_child = parent_q[safe_parent_idx]
        parent_r_per_child = parent_r[safe_parent_idx]
        parent_c_per_child = parent_c[safe_parent_idx]

        # per-child edge coefficients (num_nodes x D) and constants (num_nodes)
        q_base = self._edge_q_base[:num]
        r_base = self._edge_r_base[:num]
        c_base = self._edge_c_base[:num]

        q_ins = self._edge_q_insert[:num]
        r_ins = self._edge_r_insert[:num]
        c_ins = self._edge_c_insert[:num]

        # counts
        counts_parent = counts[safe_parent_idx]
        counts_child = counts

        # children count per parent -> per child
        children_count_per_parent = children_count[safe_parent_idx]
        # convert to float and avoid zeros
        children_count_per_parent = children_count_per_parent.to(dtype=dtype)
        children_count_per_parent = children_count_per_parent.clamp(min=1.0)

        # Compute corrections and totals in batch
        ci_plus = (counts_child + 1.0).unsqueeze(1)
        ci = counts_child.unsqueeze(1)

        corr_q = ci_plus * q_ins - ci * q_base
        corr_r = ci_plus * r_ins - ci * r_base
        corr_c = (counts_child + 1.0) * c_ins - counts_child * c_base

        denom = (counts_parent + 1.0).unsqueeze(1) * children_count_per_parent.unsqueeze(1)
        denom = denom.clamp(min=1e-12)

        total_q = (parent_q_per_child + corr_q) / denom
        total_r = (parent_r_per_child + corr_r) / denom
        total_c = (parent_c_per_child + corr_c) / denom.squeeze(1)

        # Evaluate quadratic and linear terms via batched matmul
        # q_term = sum(total_q * x_sq, dim=1)
        q_term = torch.matmul(total_q, x_sq)
        r_term = torch.matmul(total_r, x)

        scores = q_term + r_term + total_c

        # For root entries (where original parent_idx < 0), override with
        # a "newiness" approximation from stored Gaussian log-probs
        if root_mask.any():
            node_means = self._node_means[:num]
            node_vars = self._node_vars[:num].clamp(min=1e-12)
            diff_sq = (x.unsqueeze(0) - node_means) ** 2
            log_var_sum = torch.log(node_vars).sum(dim=1)
            sq_term = (diff_sq / node_vars).sum(dim=1)
            node_log_probs = -0.5 * (log_var_sum + sq_term)
            node_log_probs = node_log_probs.to(dtype=scores.dtype, device=scores.device)
            scores[root_mask] = (-node_log_probs)[root_mask]

        return scores

    def vectorized_pu_new_scores(self, instance):
        """
        Compute vectorized `pu_new(parent)` for every parent index in the
        preallocated buffers. This estimates the partition-utility of creating
        a new child under each parent containing only `instance`.

        Returns a 1-D tensor `P_parent` of length `num_nodes` where entry p is
        the utility of creating a new child under parent p.
        """
        num_nodes = len(self.idx_to_node)
        if num_nodes == 0:
            return torch.empty(0, device=self.device)

        dtype = self._path_matrix.dtype
        device = self._path_matrix.device

        x = instance.to(device=device, dtype=dtype)
        x_sq = x * x

        # Parent aggregated base sums (per parent index)
        parent_q = self._parent_base_q_sum[:num_nodes]
        parent_r = self._parent_base_r_sum[:num_nodes]
        parent_c = self._parent_base_c_sum[:num_nodes]

        counts = self._node_counts[:num_nodes].to(dtype=dtype)
        means = self._node_means[:num_nodes]
        vars = self._node_vars[:num_nodes].clamp(min=1e-12)

        children_count = self._children_count[:num_nodes].to(dtype=dtype)

        # For a brand-new child (not existing before insertion) we treat
        # Ci = 0 (no prior child). After creating it and inserting the
        # instance it will have one sample. We use the prior variance for
        # the child's variance placeholder.
        Ci = 0.0
        one = torch.tensor(1.0, dtype=dtype, device=device)

        Cp = counts

        # parent params
        beta_p = 1.0 / (Cp + 1.0)
        alpha_p = (Cp / (Cp + 1.0)).unsqueeze(1) * means

        # hypothetical child params (before insertion Ci=0 -> beta_i=1, alpha_i=0)
        beta_i = 1.0 / (Ci + 1.0)
        # alpha_i is zero because Ci=0
        alpha_i = torch.zeros_like(means)

        # Use instance as child mean, and prior_var as child var
        mu_i = x.unsqueeze(0).expand(num_nodes, -1)
        prior_val = (self.prior_var.item() if isinstance(self.prior_var, torch.Tensor)
                     else float(self.prior_var))
        var_i = torch.full((num_nodes, means.size(1)), prior_val, dtype=dtype, device=device)

        mu_diff = mu_i - alpha_p

        # compute q_ins, r_ins, c_ins for the hypothetical new child
        delta_beta = (beta_i - beta_p).unsqueeze(1)
        delta_alpha = (alpha_i - alpha_p)

        q_ins = 0.5 * (delta_beta * delta_beta) / vars
        r_ins = 0.5 * (2.0 * delta_alpha * delta_beta) / vars
        c_ins_vec = 0.5 * (var_i / vars + (delta_alpha * delta_alpha) / vars + torch.log(vars) - torch.log(var_i) - 1.0)
        c_ins = c_ins_vec.sum(dim=1)

        # For a new child corr terms equal (Ci+1)*q_ins - Ci*q_base with Ci=0
        corr_q = q_ins.squeeze(1)
        corr_r = r_ins.squeeze(1)
        corr_c = (Ci + 1.0) * c_ins - Ci * 0.0

        # Denominator: parent counts increment by 1 and children_count increments by 1
        denom = (Cp + 1.0) * (children_count + 1.0)
        denom = denom.clamp(min=1e-12).unsqueeze(1)

        # parent per-child aggregated sums (parent_q is already sum over children of c_j * q_base)
        # but shapes: parent_q (num_nodes x D), corr_q (num_nodes x D)
        total_q = (parent_q + corr_q) / denom
        total_r = (parent_r + corr_r) / denom

        # total_c is scalar per parent
        total_c = (parent_c + corr_c) / denom.squeeze(1)

        # evaluate quadratic and linear forms
        q_term = torch.matmul(total_q, x_sq)
        r_term = torch.matmul(total_r, x)

        scores = q_term + r_term + total_c

        # For parents that are effectively roots or degenerate, fall back to
        # negative log-prob computed from parent's Gaussian
        # (this is defensive -- above computation should handle parents)
        node_means = means
        node_vars = vars
        diff_sq = (x.unsqueeze(0) - node_means) ** 2
        log_var_sum = torch.log(node_vars).sum(dim=1)
        sq_term = (diff_sq / node_vars).sum(dim=1)
        node_log_probs = -0.5 * (log_var_sum + sq_term)
        node_log_probs = node_log_probs.to(dtype=scores.dtype, device=scores.device)

        # If a parent has zero children and parent sums are zero, the above formula
        # still gives reasonable result; we keep the log-prob fallback for safety
        return scores

    def _approx_cobweb_no_merge_split(self, instance):
        """
        Approximate the Cobweb DFS path (merge_split=False) using vectorized
        precomputed coefficients and matrix multiplications.

        Returns the selected node (the node the DFS would arrive at or the
        parent where a new child should be created).
        """
        num_nodes = len(self.idx_to_node)
        if num_nodes == 0:
            return self.root

        dtype = self._path_matrix.dtype
        device = self._path_matrix.device

        S_child = self.vectorized_pu_insert_scores(instance)
        P_parent = self.vectorized_pu_new_scores(instance)

        parent_idx = self._parent_idx[:num_nodes].to(dtype=torch.long, device=device)

        # sanitize S_child and P_parent for NaN/Inf (avoid device-side asserts)
        S_child = torch.nan_to_num(S_child, nan=-1e9, neginf=-1e9, posinf=1e9)
        P_parent = torch.nan_to_num(P_parent, nan=1e9, neginf=-1e9, posinf=1e9)

        # Make a safe parent index (map -1 roots to 0) for indexed ops
        parent_idx_safe = parent_idx.clone()
        if parent_idx_safe.numel() > 0:
            root_mask_idx = parent_idx_safe < 0
            if root_mask_idx.any():
                parent_idx_safe[root_mask_idx] = 0

        # Compute per-parent best child score (segment-wise max) using safe indices.
        # Use scatter_reduce if available; otherwise do a grouped reduction on CPU-safe unique parents.
        best_score_parent = torch.full((num_nodes,), float('-inf'), dtype=dtype, device=device)
        best_score_parent = best_score_parent.scatter_reduce(0, parent_idx_safe, S_child, reduce='amax')

        # Replace any non-finite best scores with a large negative finite value
        neg_large = torch.tensor(-1e9, dtype=dtype, device=device)
        best_score_parent = torch.where(torch.isfinite(best_score_parent), best_score_parent, neg_large)

        # delta per parent: best child score - new child score
        delta_parent = best_score_parent - P_parent

        # boolean mask for best child per parent (use isclose for float stability)
        valid_parent_mask = parent_idx >= 0
        parent_idx_safe = parent_idx.clone()
        parent_idx_safe[~valid_parent_mask] = 0

        best_parent_score_for_child = best_score_parent[parent_idx_safe]
        is_best_child = torch.isclose(S_child, best_parent_score_for_child, atol=1e-6) & valid_parent_mask

        # V_raw: contribution per node (non-zero only for the selected child of each parent)
        delta_for_child = delta_parent[parent_idx_safe]
        V_raw = delta_for_child * is_best_child.to(dtype=dtype)
        V_pos = V_raw.clamp(min=0.0)

        # Path average from stored (normalized) path matrix
        pm = self._path_matrix[:num_nodes, :num_nodes]
        path_avg = pm.matmul(V_pos.unsqueeze(1)).squeeze(1)

        # convert average to sum by multiplying with path lengths
        path_lengths = (pm > 0).sum(dim=1).to(dtype=dtype, device=device).clamp(min=1.0)
        path_scores = path_avg * path_lengths

        # sanitize path_scores for NaN/Inf before argmax
        path_scores = torch.nan_to_num(path_scores, nan=-1e9, neginf=-1e9, posinf=1e9)
        best_path_idx = int(torch.argmax(path_scores).item())

        # Extract path indices and inspect deltas top-down to find where DFS stops
        path_indices = self._node_to_path_indices.get(best_path_idx, [best_path_idx])

        # Walk the path in order: for each parent in the path, check delta_parent.
        termination_parent = None
        for p_idx in path_indices:
            # p_idx is the index of a node along the path; if it's a parent, check its delta
            if p_idx < 0 or p_idx >= num_nodes:
                continue
            d = float(delta_parent[p_idx].item())
            if d <= 0.0:
                termination_parent = p_idx
                break

        if termination_parent is not None:
            return self.idx_to_node[termination_parent]

        # Otherwise, DFS would fully descend to the leaf represented by best_path_idx
        final_idx = path_indices[-1]
        return self.idx_to_node[final_idx]