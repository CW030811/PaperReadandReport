# Comparison Table

| Paper | Task | Label Type | Dataset | Encoder | Decoder | Output | Loss | Inference | Novelty | Weakness | Relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Li et al. (ICLR 2026) | MILP constraint reduction for solver acceleration | Critical tight constraints derived from optimal solutions | Large-scale MILP domains; exact families uncertain in quick scan | Multi-modal MILP representation combining instance-level and abstract-level structure | Constraint prediction head | Critical-constraint predictions | Uncertain; likely supervised constraint prediction | Reduce predicted inequalities to equalities, then solve reduced MILP | Shifts reduction from variables to constraints | Exact loss and dataset composition need deeper reading | Very high for GNN/representation learning on MILP |
