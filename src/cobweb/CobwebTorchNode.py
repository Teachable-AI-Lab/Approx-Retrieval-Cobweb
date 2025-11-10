import json
import math
from random import random
import uuid
import time
from src.utils.constants import COBWEB_GREEDY_MODE
import torch

class CobwebTorchNode(object):
    """
    A CobwebNode represents a concept within the knoweldge base of a particular
    :class:`CobwebTree`. Each node contains a probability table that can be
    used to calculate the probability of different attributes given the concept
    that the node represents.

    In general the :meth:`CobwebTree.ifit`, :meth:`CobwebTree.categorize`
    functions should be used to initially interface with the Cobweb knowledge
    base and then the returned concept can be used to calculate probabilities
    of certain attributes or determine concept labels.

    This constructor creates a CobwebNode with default values. It can also be
    used as a copy constructor to "deepcopy" a node, including all references
    to other parts of the original node's CobwebTree.

    :param otherNode: Another concept node to deepcopy.
    :type otherNode: CobwebNode
    """
    # a counter used to generate unique concept names.
    _counter = 0

    def __init__(self, shape, device=None, otherNode=None):
        """Create a new CobwebNode"""
        self.concept_id = self.gensym()
        self.count = torch.tensor(0.0, dtype=torch.float, device=device,
                                  requires_grad=False)
        self.mean = torch.zeros(shape, dtype=torch.float, device=device,
                                requires_grad=False)
        self.meanSq = torch.zeros(shape, dtype=torch.float, device=device,
                                  requires_grad=False)

        self.id = str(uuid.uuid4()) # A MONKEYPATCH WE MADE
        self.timestamp = time.time() # A MONKEYPATCH WE MADE

        self.children = []
        self.parent = None
        self.tree = None
        self.sentence_id = []  # Used to track the sentence ID for this node

        if otherNode:
            self.tree = otherNode.tree
            self.parent = otherNode.parent
            self.update_counts_from_node(otherNode) 

            for child in otherNode.children:
                self.children.append(CobwebTorchNode(shape=self.tree.shape, device=self.tree.device, otherNode=child))

    def increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance.

        :param instance: A new instances to incorporate into the node.
        :type instance: :ref:`Instance<instance-rep>`
        """
        self.count += 1
        delta = instance - self.mean
        self.mean += delta / self.count
        self.meanSq += delta * (instance - self.mean)

    def update_counts_from_node(self, other):
        """
        Increments the counts of the current node by the amount in the
        specified node.

        This function is used as part of copying nodes and in merging nodes.

        :param node: Another node from the same CobwebTree
        :type node: CobwebNode
        """
        delta = other.mean - self.mean
        self.meanSq = (self.meanSq + other.meanSq + delta * delta *
                       ((self.count * other.count) / (self.count + other.count)))
        self.mean = ((self.count * self.mean + other.count * other.mean) /
                     (self.count + other.count))
        self.count += other.count

    @property
    def std(self):
        return torch.sqrt(self.var)

    @property
    def var(self):
        return self.tree.compute_var(self.meanSq, self.count)

    def log_prob_class_given_instance(self, instance):
        log_prob = self.log_prob(instance)
        log_prob += torch.log(self.count) - torch.log(self.tree.root.count)
        return log_prob

    def log_prob(self, instance):
        var = self.var
        log_prob = -(0.5 * torch.log(var) + 0.5 * torch.log(2 * self.tree.pi_tensor) +
                     0.5 * torch.square(instance - self.mean) / var).sum()
        return log_prob

    def score_insert(self, instance):
        """
        Returns the score that would result from inserting the instance into
        the current node.

        This operation can be used instead of inplace and copying because it
        only looks at the attr values used in the instance and reduces iteration.
        """
        count = self.count + 1
        delta = instance - self.mean
        mean = self.mean + delta / count
        meanSq = self.meanSq + delta * (instance - mean)

        var = self.tree.compute_var(meanSq, count)
        std = torch.sqrt(var)

        if (self.tree.use_info):
            score = (0.25 * torch.log(var)).sum()
        else:
            score = -(1 / (2 * torch.sqrt(self.tree.pi_tensor) * std)).sum()

        return score

    def score_merge(self, other, instance):
        """
        Returns the expected correct guesses that would result from merging the
        current concept with the other concept and inserting the instance.

        This operation can be used instead of inplace and copying because it
        only looks at the attr values used in the instance and reduces iteration.
        """
        delta = other.mean - self.mean
        meanSq = (self.meanSq + other.meanSq + delta * delta *
                  ((self.count * other.count) / (self.count + other.count)))
        mean = ((self.count * self.mean + other.count * other.mean) /
                (self.count + other.count))
        count = self.count + other.count

        count = count + 1
        delta = instance - mean
        mean += delta / count
        meanSq += delta * (instance - mean)

        var = self.tree.compute_var(meanSq, count)

        if (self.tree.use_info):
            score = (0.5 * torch.log(var)).sum()
        else:
            score = -(1 / (2 * torch.sqrt(self.tree.pi_tensor) * torch.sqrt(var))).sum()

        return score

    def get_basic(self):
        """
        Climbs up the tree from the current node (probably a leaf),
        computes the category utility score, and returns the node with
        the highest score.
        """
        curr = self
        best = self
        best_cu = self.category_utility()

        while curr.parent:
            curr = curr.parent
            curr_cu = curr.category_utility()
            if curr_cu > best_cu:
                best = curr
                best_cu = curr_cu

        return best

    def get_best(self, instance):
        """
        Climbs up the tree from the current node (probably a leaf),
        computes the category utility score, and returns the node with
        the highest score.
        """
        curr = self
        best = self
        best_ll = self.log_prob_class_given_instance(instance)

        while curr.parent:
            curr = curr.parent
            curr_ll = curr.log_prob_class_given_instance(instance)
            if curr_ll > best_ll:
                best = curr
                best_ll = curr_ll

        return best

    def category_utility(self):
        p_of_c = self.count / self.tree.root.count
        root_mean, root_var = self.tree.root.mean_var()
        curr_mean, curr_var = self.mean_var()

        return p_of_c * self.tree.compute_score(curr_mean, curr_var,
                                                root_mean, root_var)

    def mean_var_new(self, instance):
        var = torch.zeros(self.tree.shape, dtype=torch.float,
                          device=self.tree.device)
        var += self.tree.prior_var

        return instance, var

    def mean_var(self):
        return self.mean, self.var

    def mean_var_insert(self, instance):
        count = self.count + 1
        delta = instance - self.mean
        mean = self.mean + delta / count
        meanSq = self.meanSq + delta * (instance - mean)

        var = self.tree.compute_var(meanSq, count)

        return mean, var

    def mean_var_merge(self, other, instance):
        delta = other.mean - self.mean
        meanSq = (self.meanSq + other.meanSq + delta * delta *
                  ((self.count * other.count) / (self.count + other.count)))
        mean = ((self.count * self.mean + other.count * other.mean) /
                (self.count + other.count))
        count = self.count + other.count

        count = count + 1
        delta = instance - mean
        mean += delta / count
        meanSq += delta * (instance - mean)

        var = self.tree.compute_var(meanSq, count)

        return mean, var

    def partition_utility(self):
        """
        Return the category utility of a particular division of a concept into
        its children.

        Category utility is always calculated in reference to a parent node and
        its own children. This is used as the heuristic to guide the concept
        formation process. Category Utility is calculated as:

        .. math::

            CU(\\{C_1, C_2, \\cdots, C_n\\}) = \\frac{1}{n} \\sum_{k=1}^n
            P(C_k) \\left[ \\sum_i \\sum_j P(A_i = V_{ij} | C_k)^2 \\right] -
            \\sum_i \\sum_j P(A_i = V_{ij})^2

        where :math:`n` is the numer of children concepts to the current node,
        :math:`P(C_k)` is the probability of a concept given the current node,
        :math:`P(A_i = V_{ij} | C_k)` is the probability of a particular
        attribute value given the concept :math:`C_k`, and :math:`P(A_i =
        V_{ij})` is the probability of a particular attribute value given the
        current node.

        In general this is used as an internal function of the cobweb algorithm
        but there may be times when it would be useful to call outside of the
        algorithm itself.

        :return: The category utility of the current node with respect to its
                 children.
        :rtype: float
        """
        if len(self.children) == 0:
            return 0.0

        score = 0.0
        parent_mean, parent_var = self.mean_var()

        for child in self.children:
            p_of_child = child.count / self.count
            child_mean, child_var = child.mean_var()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          parent_mean,
                                                          parent_var)

        return score / len(self.children)

    def get_best_operation(self, instance, best1, best2, best1_pu):
        """
        Given an instance, the two best children based on category utility and
        a set of possible operations, find the operation that produces the
        highest category utility, and then return the category utility and name
        for the best operation. In the case of ties, an operator is randomly
        chosen.

        Given the following starting tree the results of the 4 standard Cobweb
        operations are shown below:

        .. image:: images/Original.png
            :width: 200px
            :align: center

        * **Best** - Categorize the instance to child with the best category
          utility. This results in a recurisve call to :meth:`cobweb
          <concept_formation.cobweb.CobwebTree.cobweb>`.

            .. image:: images/Best.png
                :width: 200px
                :align: center

        * **New** - Create a new child node to the current node and add the
          instance there. See: :meth:`create_new_child
          <concept_formation.cobweb.CobwebNode.create_new_child>`.

            .. image:: images/New.png
                :width: 200px
                :align: center

        * **Merge** - Take the two best children, create a new node as their
          mutual parent and add the instance there. See: :meth:`merge
          <concept_formation.cobweb.CobwebNode.merge>`.

            .. image:: images/Merge.png
                    :width: 200px
                    :align: center

        * **Split** - Take the best node and promote its children to be
          children of the current node and recurse on the current node. See:
          :meth:`split <concept_formation.cobweb.CobwebNode.split>`

            .. image:: images/Split.png
                :width: 200px
                :align: center

        Each operation is entertained and the resultant category utility is
        used to pick which operation to perform. The list of operations to
        entertain can be controlled with the possible_ops parameter. For
        example, when performing categorization without modifying knoweldge
        only the best and new operators are used.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :param best1: A tuple containing the relative cu of the best child and
            the child itself, as determined by
            :meth:`CobwebNode.two_best_children`.
        :type best1: (float, CobwebNode)
        :param best2: A tuple containing the relative cu of the second best
            child and the child itself, as determined by
            :meth:`CobwebNode.two_best_children`.
        :type best2: (float, CobwebNode)
        :param possible_ops: A list of operations from ["best", "new", "merge",
            "split"] to entertain.
        :type possible_ops: ["best", "new", "merge", "split"]
        :return: A tuple of the category utility of the best operation and the
            name of the best operation.
        :rtype: (cu_bestOp, name_bestOp)
        """
        if not best1:
            raise ValueError("Need at least one best child.")

        operations = []

        operations.append((best1_pu, random(), "best"))
        operations.append((self.pu_for_new_child(instance), random(), 'new'))
        if len(self.children) > 2 and best2:
            operations.append((self.pu_for_merge(best1, best2, instance),
                               random(), 'merge'))
        if len(best1.children) > 0:
            operations.append((self.pu_for_split(best1), random(), 'split'))

        operations.sort(reverse=True)
        best_op = (operations[0][0], operations[0][2])
        return best_op

    def two_best_children(self, instance):
        """
        Calculates the category utility of inserting the instance into each of
        this node's children and returns the best two. In the event of ties
        children are sorted first by category utility, then by their size, then
        by a random value.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility and indices for the two best children
            (the second tuple will be ``None`` if there is only 1 child).
        :rtype: ((cu_best1,index_best1),(cu_best2,index_best2))
        """
        if len(self.children) == 0:
            raise Exception("No children!")

        relative_pus = []
        parent_mean, parent_var = self.mean_var_insert(instance)

        for child in self.children:
            p_of_c = (child.count + 1) / (self.count + 1)
            mean, var = child.mean_var_insert(instance)
            score_gain = p_of_c * self.tree.compute_score(mean, var,
                                                          parent_mean,
                                                          parent_var)

            p_of_c = child.count / (self.count + 1)
            mean, var = child.mean_var()
            score_gain -= p_of_c * self.tree.compute_score(mean, var,
                                                           parent_mean,
                                                           parent_var)

            relative_pus.append((score_gain, child.count, random(), child))

        relative_pus.sort(reverse=True)

        best1 = relative_pus[0][3]
        if COBWEB_GREEDY_MODE:
            best1_pu = 0
        else:
            best1_pu = self.pu_for_insert(best1, instance)

        best2 = None
        if len(relative_pus) > 1:
            best2 = relative_pus[1][3]

        return best1_pu, best1, best2

    def pu_for_insert(self, child, instance):
        """
        Compute the category utility of adding the instance to the specified
        child.

        This operation does not actually insert the instance into the child it
        only calculates what the result of the insertion would be. For the
        actual insertion function see: :meth:`CobwebNode.increment_counts` This
        is the function used to determine the best children for each of the
        other operations.

        :param child: a child of the current node
        :type child: CobwebNode
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility of adding the instance to the given node
        :rtype: float

        .. seealso:: :meth:`CobwebNode.two_best_children` and
            :meth:`CobwebNode.get_best_operation`

        """
        score = 0.0
        parent_mean, parent_var = self.mean_var_insert(instance)

        for c in self.children:
            if c == child:
                p_of_child = (c.count + 1) / (self.count + 1)
                child_mean, child_var = c.mean_var_insert(instance)
            else:
                p_of_child = (c.count) / (self.count + 1)
                child_mean, child_var = c.mean_var()

            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          parent_mean,
                                                          parent_var)

        return score / len(self.children)

    def create_new_child(self, instance):
        """
        Create a new child (to the current node) with the counts initialized by
        the *given instance*.

        This is the operation used for creating a new child to a node and
        adding the instance to it.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: The new child
        :rtype: CobwebNode
        """
        new_child = CobwebTorchNode(shape=self.tree.shape, device=self.tree.device)
        new_child.parent = self
        new_child.tree = self.tree
        new_child.increment_counts(instance)
        self.children.append(new_child)
        return new_child

    def pu_for_new_child(self, instance):
        """
        Return the category utility for creating a new child using the
        particular instance.

        This operation does not actually create the child it only calculates
        what the result of creating it would be. For the actual new function
        see: :meth:`CobwebNode.create_new_child`.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility of adding the instance to a new child.
        :rtype: float

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        score = 0.0
        parent_mean, parent_var = self.mean_var_insert(instance)

        for c in self.children:
            p_of_child = c.count / (self.count + 1)
            child_mean, child_var = c.mean_var()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          parent_mean,
                                                          parent_var)

        # score for new
        p_of_child = 1.0 / (self.count + 1)
        child_mean, child_var = self.mean_var_new(instance)
        score += p_of_child * self.tree.compute_score(child_mean, child_var,
                                                      parent_mean, parent_var)

        return score / (len(self.children) + 1)

    def merge(self, best1, best2):
        """
        Merge the two specified nodes.

        A merge operation introduces a new node to be the merger of the the two
        given nodes. This new node becomes a child of the current node and the
        two given nodes become children of the new node.

        :param best1: The child of the current node with the best category
            utility
        :type best1: CobwebNode
        :param best2: The child of the current node with the second best
            category utility
        :type best2: CobwebNode
        :return: The new child node that was created by the merge
        :rtype: CobwebNode
        """
        new_child = CobwebTorchNode(shape=self.tree.shape, device=self.tree.device)
        new_child.parent = self
        new_child.tree = self.tree

        new_child.update_counts_from_node(best1)
        new_child.update_counts_from_node(best2)
        best1.parent = new_child
        best2.parent = new_child
        new_child.children.append(best1)
        new_child.children.append(best2)
        self.children.remove(best1)
        self.children.remove(best2)
        self.children.append(new_child)

        return new_child

    def pu_for_merge(self, best1, best2, instance):
        """
        Return the category utility for merging the two best children.

        This does not actually merge the two children it only calculates what
        the result of the merge would be. For the actual merge operation see:
        :meth:`CobwebNode.merge`

        :param best1: The child of the current node with the best category
            utility
        :type best1: CobwebNode
        :param best2: The child of the current node with the second best
            category utility
        :type best2: CobwebNode
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: The category utility that would result from merging best1 and
            best2.
        :rtype: float

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        score = 0.0
        parent_mean, parent_var = self.mean_var_insert(instance)

        for c in self.children:
            if c == best1 or c == best2:
                continue

            p_of_child = c.count / (self.count + 1)
            child_mean, child_var = c.mean_var()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          parent_mean,
                                                          parent_var)

        p_of_child = (best1.count + best2.count + 1) / (self.count + 1)
        child_mean, child_var = best1.mean_var_merge(best2, instance)
        score += p_of_child * self.tree.compute_score(child_mean, child_var,
                                                      parent_mean, parent_var)

        return score / (len(self.children) - 1)

    def split(self, best):
        """
        Split the best node and promote its children

        A split operation removes a child node and promotes its children to be
        children of the current node. Split operations result in a recursive
        call of cobweb on the current node so this function does not return
        anything.

        :param best: The child node to be split
        :type best: CobwebNode
        """
        self.children.remove(best)
        for child in best.children:
            child.parent = self
            child.tree = self.tree
            self.children.append(child)

    def pu_for_split(self, best):
        """
        Return the category utility for splitting the best child.

        This does not actually split the child it only calculates what the
        result of the split would be. For the actual split operation see:
        :meth:`CobwebNode.split`. Unlike the category utility calculations for
        the other operations split does not need the instance because splits
        trigger a recursive call on the current node.

        :param best: The child of the current node with the best category
            utility
        :type best: CobwebNode
        :return: The category utility that would result from splitting best
        :rtype: float

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        score = 0.0
        parent_mean, parent_var = self.mean_var()

        for c in self.children:
            if c == best:
                continue
            p_of_child = c.count / self.count
            child_mean, child_var = c.mean_var()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          parent_mean,
                                                          parent_var)

        for c in best.children:
            p_of_child = c.count / self.count
            child_mean, child_var = c.mean_var()
            score += p_of_child * self.tree.compute_score(child_mean,
                                                          child_var,
                                                          parent_mean,
                                                          parent_var)

        return (score / (len(self.children) - 1 + len(best.children)))

    def is_exact_match(self, instance):
        """
        Returns true if the concept exactly matches the instance.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: whether the instance perfectly matches the concept
        :rtype: boolean

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        std = torch.sqrt(self.meanSq / self.count)
        if not torch.isclose(std, torch.zeros(std.shape,device=self.tree.device)).all():
            return False
        return torch.isclose(instance, self.mean).all()

    def __hash__(self):
        """
        The basic hash function. This hashes the concept name, which is
        generated to be unique across concepts.
        """
        return hash("CobwebNode" + str(self.concept_id))

    def gensym(self):
        """
        Generate a unique id and increment the class _counter.

        This is used to create a unique name for every concept. As long as the
        class _counter variable is never externally altered these keys will
        remain unique.

        """
        self.__class__._counter += 1
        return self.__class__._counter

    def __str__(self):
        """
        Call :meth:`CobwebNode.pretty_print`
        """
        return self.pretty_print()

    def pretty_print(self, depth=0):
        """
        Print the categorization tree

        The string formatting inserts tab characters to align child nodes of
        the same depth.

        :param depth: The current depth in the print, intended to be called
                      recursively
        :type depth: int
        :return: a formated string displaying the tree and its children
        :rtype: str
        """
        ret = str(('\t' * depth) + "|-" + str(self.mean) + ":" +
                  str(self.count) + '\n')

        for c in self.children:
            ret += c.pretty_print(depth+1)

        return ret

    def depth(self):
        """
        Returns the depth of the current node in its tree

        :return: the depth of the current node in its tree
        :rtype: int
        """
        if self.parent:
            return 1 + self.parent.depth()
        return 0

    def num_concepts(self):
        """
        Return the number of concepts contained below the current node in the
        tree.

        When called on the :attr:`CobwebTree.root` this is the number of nodes
        in the whole tree.

        :return: the number of concepts below this concept.
        :rtype: int
        """
        children_count = 0
        for c in self.children:
            children_count += c.num_concepts()
        return 1 + children_count

    def iterative_output_json_helper(self):
        output = {}
        output['count'] = self.count.item()
        output['mean'] = self.mean.tolist()
        output['meanSq'] = self.meanSq.tolist()
        output['sentence_id'] = self.sentence_id if hasattr(self, 'sentence_id') else None
        return json.dumps(output)

    def iterative_output_json(self):
        output = ""

        visited = set()
        curr = self

        while curr is not None:
            if curr.concept_id not in visited:
                node_output = curr.iterative_output_json_helper()
                if len(output) > 1 and output[-1] == "}":
                    output += ", "
                output += node_output[:-1]
                output += ', "children": ['
                visited.add(curr.concept_id)

            for child in curr.children:
                if child.concept_id not in visited:
                    curr = child
                    break
            else:
                curr = curr.parent
                output += "]}"

        return output

    # TODO remove and just use above output, left for legacy use with viz.
    def output_json(self):
        return json.dumps(self.output_dict())

    def visualize(self):
        from matplotlib import pyplot as plt
        plt.imshow(self.mean.numpy())
        plt.show()

    def output_dict(self):
        """
        Outputs the categorization tree in JSON form

        :return: an object that contains all of the structural information of
                 the node and its children
        :rtype: obj
        """
        output = {}
        output['name'] = "Concept" + str(self.concept_id)
        output['size'] = self.count.item()
        output['children'] = []

        temp = {}
        temp['_category_utility'] = {"#ContinuousValue#": {'mean': self.category_utility().item(), 'std': 1, 'n': 1}}
        for child in self.children:
            output["children"].append(child.output_dict())

        output['counts'] = temp
        output['mean'] = self.mean.tolist()
        output['meanSq'] = self.meanSq.tolist()

        return output