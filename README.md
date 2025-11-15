# Approximate Retrieval with Cobweb

## Motivations

The goal is to generalize Cobweb's exact retrieval methods in hybrid with existing retrieval to produce better retrieval. Our contributions are listed below:
*   We will analyze different hybrids of Cobweb's graph-based search and dense search retrieval variants to see what works best.
*   We will analyze the progression of BFS in comparison to DFS, with the goal of comparing how a best-first node expansion 
*   We will analyze Cobweb as a suitable replacement to current product-quantization solutions

## Implementation Strategy: Approximate Cobweb

*   We use ```ApproxCobwebWrapper.py``` - a modified CobwebWrapper class that builds different indices depending on the first method and the second method used in the hybrid system.
    *   We'll implement like this for now and then depending on the strategies that work and don't work, 
    *   Indexes will need to be built for each transition node in terms of only their respective leaves
*   Once we've identified which combination of methods functions the best, we'll define a new implementation that spurs time complexity for optimal conditions
*   Implementation is as follows:
    *   We collect and build the general index for dot-product / pathsum
    *   We collect all indexes of the leaves under each transition node
    *   We can then do a spliced calculation (only calculating for nodes we're interested in) and take the average based on that

## Implementation Strategy: Cobweb for PQ

Traditional Product-Quantization operates as follows:
*   The idea is as follows: (to go from dim N to dim K)
*   Split a high-degree vector up into K subspaces of dimension N/K
*   Train a kNN on each subspace such that it produces a number of centroids
*   To encode a vector:
    *   Find the nearest centroid in each of the K subspaces
    *   Create a vector with the centroid index of each of the K subspaces

There are many different ways we can choose to implement PQ with Cobweb!
*   One way is taking the path of a given vector as its new encoding - however similar vectors don't always have the same path so this might be chopped

## Speeding up Cobweb

*   The core of the idea revolves around our pathsum calculation - we can just pathsum to figure out where the node goes, and then update the heuristics of the path of the node
*   The most computationally expensive part of the fitting process for Cobweb is the decision-making at each node. So even though we may not recognize a time-complex advantage, we will definitely be able to find a computational advantage to updating in retrospect vs. calculating in real time!
*   Implementation plan:
    *   Implement fitting in the way that categorize does fitting (need to verify that this works first)
        *   We should implement the categorize function within ifit to find the necessary node and then traverse backwards
        *   Store the nodes's parents for easy backwards traversal, update each node as you go up the path once identifying categorization through increment_counts() (only time we need to call "create_new_child" is at the node that we initialize our new node)
        *   Need to verify that categorization will actually sort to intermediary nodes if we provide it with the option to do so
        *   Additionally, we need to flesh out fringe split and other edge cases well so that 
    *   Initialize all the structures needed for computation of Pathsum within the CobwebTree itself
    *   Simply change the categorization logic to PathSum logic
*   Note that we may even be able to bring back a whole-tree merge split pass-through as we traverse back up the tree

## Benchmarks

*   Methods tested against:
    *   HNSWLib
    *   Annoy (Spotify)
    *   ScaNN

*   Datasets we want to test on:
    *   MS-MARCO

*   Metrics used