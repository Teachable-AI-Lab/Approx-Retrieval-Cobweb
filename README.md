# Approximate Retrieval with Cobweb

# Motivations

The goal is to generalize Cobweb's exact retrieval methods in hybrid with existing retrieval to produce better retrieval. Our contributions are listed below:
*   We will analyze different hybrids of Cobweb's graph-based search and dense search retrieval variants to see what works best.
*   We will analyze the progression of BFS in comparison to DFS, with the goal of comparing how a best-first node expansion 
*   We will analyze Cobweb as a suitable replacement to current product-quantization solutions

# Implementation Strategy: Approximate Cobweb

*   We use ```ApproxCobwebWrapper.py``` - a modified CobwebWrapper class that builds different indices depending on the first method and the second method used in the hybrid system.
    *   We'll implement like this for now and then depending on the strategies that work and don't work, 
    *   Indexes will need to be built for each transition node in terms of only their respective leaves
*   Once we've identified which combination of methods functions the best, we'll define a new implementation that spurs time complexity for optimal conditions

*   Implementation for ```ApproxCobwebWrapper.py```:
    *   

# Implementation Strategy: Cobweb for PQ

*   We'll create a class that does the following:
    *   Feeds all documents into Cobweb
    *   For each document, 

## Benchmarks

*   Methods tested against:
    *   FAISS IVF-PQ
    *   HNSWLib
    *   Annoy (Spotify)
    *   ScaNN

*   Datasets we want to test on:
    *   f

*   Metrics used