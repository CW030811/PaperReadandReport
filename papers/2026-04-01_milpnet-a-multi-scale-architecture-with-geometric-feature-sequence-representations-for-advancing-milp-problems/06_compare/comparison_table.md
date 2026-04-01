# Comparison Table

| Paper | Task | Label Type | Dataset | Encoder | Decoder | Output | Loss | Inference | Novelty | Weakness | Relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Wang et al. (ICLR 2026) | MILP representation learning beyond GNNs | Supervision for feasibility, objective, and solution mappings; exact setup uncertain | Foldable MILPs plus real-world MILP benchmarks | Multi-scale hybrid attention over geometric sequences | Prediction heads for optimization-related mappings | Feasibility, objective value, and solution approximations | Uncertain in quick scan | Integrated into an end-to-end solver pipeline | Challenges GNN sufficiency for MILP representation | Exact training targets and benchmark details need deeper reading | Extremely high if you are questioning graph encoders for MILP |
