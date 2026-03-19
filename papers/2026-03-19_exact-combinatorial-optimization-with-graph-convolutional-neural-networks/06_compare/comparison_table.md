# Comparison Table

| Paper | Task | Label Type | Dataset | Encoder | Decoder | Output | Loss | Inference | Novelty | Weakness | Relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gasse et al. (2019) | MILP branch-and-bound variable selection | Strong branching expert action at each B&B node | Random instances from set covering, combinatorial auction, capacitated facility location, maximum independent set | Bipartite GCNN over variable-constraint graph | Variable-node MLP head after one bipartite graph convolution | Masked softmax over candidate branching variables | Cross-entropy behavioral cloning | Pick top-ranked branching variable inside SCIP | Graph state encoding + action classification imitation + full-solver evaluation + larger-instance generalization | Partial solver observability, surrogate objective mismatch, limited very-large-instance generalization | High; strong reference paper for learned branching in exact solvers |
