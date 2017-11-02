# slates_semisynth_expts
Semi-synthetic experiments to test several approaches for off-policy evaluation and optimization of slate recommenders.

Contact: Adith Swaminathan (adswamin@microsoft.com)

These python scripts and classes run semi-synthetic experiments on the MSLR and MQ datasets
to study off-policy estimators for the slate bandit problem (combinatorial contextual bandits).

For Evaluation experiments:
Usage: python Parallel.py
Refer Parallel.py::main for examples on how to set up other variants of experiments
The make_parallel_eval.sh bash script creates the entire suite of experiments reported in [1] as parallel cluster jobs.

Data:
MSLR-30K has 31K queries, each with up to 1251 judged documents on relevance scale of {0, 1, 2, 3, 4}
MSLR-10K has 10K queries, each with up to 908 judged documents on relevance scale of {0, 1, 2, 3, 4}
Both MSLR datasets have <query, document> features of dimension 136

MQ2007 has 1692 queries, each with between 6 and 147 documents judged on relevance scale of {0, 1, 2}
MQ2008 has 784 queries, each with between 5 and 121 documents judged on relevance scale of {0, 1, 2}
Both datasets have <query, document> feature vectors of dimension 46

Refer Datasets.py::main for how to read in these datasets
    Download and uncompress the dataset files in the ../../Data/ folder
    MSLR: https://www.microsoft.com/en-us/research/project/mslr/
    MQ: https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/
    After reading in the uncompressed datasets once, 
    the Datasets script creates *.npz files in the ../../Data/ folder as pre-processed numpy arrays for faster consumption by other scripts
    
    
For Optimization experiments:
Usage: python Optimization.py
Refer Optimization.py::main for examples on how to set up other variants of experiments
The make_parallel_opt.sh bash script creates the entire suite of experiments reported in [1] as parallel cluster jobs.

[1] Off policy evaluation for slate recommendation, https://arxiv.org/abs/1605.04812 ; https://nips.cc/Conferences/2017/Schedule?showEvent=9146