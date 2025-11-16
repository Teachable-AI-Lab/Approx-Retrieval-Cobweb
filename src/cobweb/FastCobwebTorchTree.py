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
                 use_kl=True, prior_var=None, alpha=1e-8, device=None, gradient_flow=False, precompute=40000):
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
        # Initialize unused vars to the prior variance to avoid log(0)/div0
        prior_val = (self.prior_var.item() if isinstance(self.prior_var, torch.Tensor)
                     else float(self.prior_var))
        self._node_vars = torch.full((precompute, self.shape[0]), prior_val,
                                     device=self.device, dtype=torch.float)
        self._node_to_path_indices = {} # designed to speed up path matrix computations
        self._path_matrix = torch.zeros(precompute, precompute, device=self.device, dtype=torch.float) # precompute this to be large
        self._leaf_idxs = torch.zeros(precompute, device=self.device, dtype=torch.float)

    def resize_structs(self, new_size):
        """
        Helper function to resize old structure size to new structure size!

        Only _node_means, _node_vars, and _path_matrix which should be resized as they are not
        adaptive.
        """
        def resize_tensor_2d(tensor, new_shape):
            old_rows, old_cols = tensor.shape
            new_rows, new_cols = new_shape
            result = torch.zeros(new_rows, new_cols, dtype=tensor.dtype, device=tensor.device)
            rows_to_copy = min(old_rows, new_rows)
            cols_to_copy = min(old_cols, new_cols)
            result[:rows_to_copy, :cols_to_copy] = tensor[:rows_to_copy, :cols_to_copy]
            return result
        
        self._node_means = resize_tensor_2d(self._node_means, (new_size, self.shape[0]))
        self._node_vars = resize_tensor_2d(self._node_vars, (new_size, self.shape[0]))
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

        if len(self.idx_to_node) >= len(self._node_means):
            self.resize_structs(len(self.idx_to_node) * 2 + 1)

        def increment_up(current_node):
            # traverse up from the node and increment counts
            parentUp = current_node
            while parentUp:
                parentUp.increment_counts(instance)

                # update mean/vars in preallocated buffers
                if hash(parentUp) in self.hash_to_idx:
                    idx_up = self.hash_to_idx[hash(parentUp)]
                    self._node_means[idx_up] = parentUp.mean
                    self._node_vars[idx_up] = parentUp.var

                if hasattr(parentUp, "parent") and parentUp.parent:
                    parentUp = parentUp.parent
                else:
                    parentUp = None

        current = self.soft_find_first_new_greedy(instance)

        if not current.children and (current.is_exact_match(instance) or
                                         current.count == 0):
            
            self.hash_to_idx[hash(current)] = len(self.idx_to_node)

            increment_up(current)

            # update pathsum datastructures
            idx = len(self.idx_to_node)
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
            self.idx_to_node.append(current)

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

            self.hash_to_idx[hash(newChild)] = len(self.idx_to_node)

            increment_up(newChild)

            # update pathsum datastructures
            idx = len(self.idx_to_node)
            self._node_to_path_indices[idx] = self._node_to_path_indices[self.hash_to_idx[hash(newChild.parent)]] + [idx]
            row = torch.zeros(self._path_matrix.size(1), dtype=self._path_matrix.dtype, device=self.device)
            indices = self._node_to_path_indices[idx]
            if len(indices) > 0:
                inds = torch.tensor(indices, dtype=torch.long, device=self.device)
                row[inds] = 1.0 / float(len(indices))
            self._path_matrix[idx] = row
            self.idx_to_node.append(newChild)

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

    def old_categorize(self, instance):
        """
        A cobweb specific version of categorize, not intended to be
        externally called.

        .. seealso:: :meth:`CobwebTree.categorize`
        """
        current = self.root

        while True:
            if (len(current.children) == 0):
                return current

            parent = current
            current = None
            best_score = None

            for child in parent.children:
                score = child.log_prob_class_given_instance(instance)

                if ((current is None) or ((best_score is None) or (score > best_score))):
                    best_score = score
                    current = child

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
        Matrix-only soft approximation of `find_first_new_greedy`.

        This function avoids any tree traversal and uses only the
        procedurally-updated tensors in the class (e.g. `_node_means`,
        `_node_vars`, `_path_matrix`, and `idx_to_node`) to compute a
        differentiable / matrix-based approximation of the greedy
        top-down search that would return the first node along the
        greedy path whose best action is 'new'.

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
        # Fast-path: nothing to do
        if len(self.idx_to_node) == 0:
            return self.root

        # Bring input to the same device
        x = instance.to(self.device)

        # Work only with active nodes
        num_nodes = len(self.idx_to_node)
        node_means = self._node_means[:num_nodes]
        node_vars = self._node_vars[:num_nodes]

        # Stabilize variances
        eps = 1e-12
        node_vars = node_vars.clamp(min=eps)

        # Mahalanobis-like term: (x - mu)^2 / var
        diff_sq = (x.unsqueeze(0) - node_means) ** 2
        scaled_diff = diff_sq / node_vars
        sq_term = scaled_diff.sum(dim=1)

        # Node log-probability (up to constant): -0.5*(log(var).sum + sq_term)
        log_var_sum = torch.log(node_vars).sum(dim=1)
        node_log_probs = -0.5 * (log_var_sum + sq_term)

        # Invert log-probs to get a "newiness" score: nodes that fit the
        # instance poorly (low log-prob) get higher newiness.
        newiness = -node_log_probs

        # Normalize newiness to avoid scale issues (still pure tensor ops)
        new_mean = newiness.mean()
        new_std = newiness.std(unbiased=False)
        new_std = new_std.clamp(min=eps)
        newiness = (newiness - new_mean) / new_std

        # Multiply by the path matrix to simulate top-down accumulation.
        # `_path_matrix` rows correspond to normalized path indicators for
        # each stored node; slicing matches active nodes.
        pm = self._path_matrix[:num_nodes, :num_nodes]
        # aggregated score per stored-node index (pure matmul)
        agg_scores = pm.matmul(newiness)

        # Choose the best aggregated score's index (no loops) and return node
        best_idx = int(torch.argmax(agg_scores).item())

        # Guard: if index is out-of-range for any reason, return None
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
