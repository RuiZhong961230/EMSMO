# EMSMO
Evolutionary Multi-mode Slime mold Optimization: A hyper-heuristic algorithm inspired by slime mold foraging behaviors

## Abstract
This paper proposes a novel hyper-heuristic algorithm termed evolutionary multi-mode slime mold optimization (EMSMO) for addressing continuous optimization problems. The architecture of a typical hyper-heuristic algorithm comprises two main components: the high-level component and the low-level component. The low-level component contains a set of low-level heuristics (LLHs) and intrinsic problem attributes, while the high-level component manipulates the LLHs to construct the sequence of heuristics. Inspired by the foraging behaviors of slime mold, we designed four easy-implemented search strategies including the search for food, approach food, wrap food, and re-initialization as the LLHs for the low-level component. In the high-level component, we adopt an improvement-based probabilistic selection function that contains two metrics: (1) the probability of improvement and (2) the normalized improvement. The selection function cooperates with the roulette wheel strategy to construct the optimization sequence. To evaluate the performance of our proposal, we implement comprehensive numerical experiments on CEC2013 benchmark functions and three engineering optimization problems. Six classic or advanced evolutionary algorithms and three hyper-heuristic algorithms are applied as competitor algorithms to evaluate the competitiveness of EMSMO. Experimental and statistical results show that EMSMO has broad prospects for solving continuous optimization problems.

## Citation
@article{Zhong:24,  
title={Evolutionary multi-mode slime mold optimization: a hyper-heuristic algorithm inspired by slime mold foraging behaviors},  
author={Rui Zhong and Enzhi Zhang and Masaharu Munetomo },  
journal={The Journal of Supercomputing},  
volume={80},  
pages={12186–12217},  
year={2024},  
publisher={Springer},  
doi = {https://doi.org/10.1007/s11227-024-05909-0 },  
}  

## Datasets and Libraries
CEC benchmarks are provided by the opfunu library and engineering problems are provided by the enoppy library.
