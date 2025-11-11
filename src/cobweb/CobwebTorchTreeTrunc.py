##-------------------------------------------------------------
## CobwebTorchTreeTrunc.py
## A truncated Cobweb Torch Tree for approximate retrieval.
##-------------------------------------------------------------

import json
import math
from random import shuffle, random
from math import log, isclose
from collections import defaultdict
import heapq

import torch
from src.cobweb.CobwebTorchNode import CobwebTorchNode

class CobwebTorchTreeTrunc(object):
    """
    The CobwebTree contains the knowledge base of a particular instance of the
    cobweb algorithm and can be used to fit and categorize instances.
    """

    def __init__(self, shape, use_info=True, acuity_cutoff=False,
                 use_kl=True, prior_var=None, alpha=1e-8, device=None, gradient_flow=False):
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
        self.clear()

    def clear(self):
        """
        Clears the concepts of the tree and resets the node map.
        """
        self.root = CobwebTorchNode(shape=self.shape, device=self.device, )
        self.root.tree = self
        # Build node_map for id->node lookup
        self.node_map = {self.root.id: self.root}
        

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

    def ifit(self, instance):
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
            return self.cobweb(instance)

        with torch.no_grad():
            return self.cobweb(instance)

    def cobweb(self, instance):
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
                # print("leaf match")
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
                                                            best2, best1_pu)

                # print(best_action)
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
                        add.append((-child_score, score, random(), c))
                    else:
                        heapq.heappush(queue, (-child_score, score, random(), c))

                if greedy:
                    add.sort()  # sort by neg_score
                    queue.extend(add[::-1])

        if retrieve_k is None:
            return best if use_best else curr
        return [retrieved[i][-1] for i in range(retrieve_k)]

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