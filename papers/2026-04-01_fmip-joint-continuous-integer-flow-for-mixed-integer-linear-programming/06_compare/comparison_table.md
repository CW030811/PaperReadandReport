# Comparison Table

| Paper | Task | Label Type | Dataset | Encoder | Decoder | Output | Loss | Inference | Novelty | Weakness | Relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Li et al. (ICLR 2026) | Generative MILP heuristic learning | MILP solution distributions over integer and continuous variables | Eight standard MILP benchmarks; exact families uncertain in quick scan | Backbone-agnostic joint flow framework | Conditional flow matching generator with guidance | Full continuous-integer solution | Flow-matching-based; exact formulation needs deeper read | Guided iterative solution refinement before downstream solving | First joint continuous-integer generative MILP framework | Exact loss and solver integration details need closer reading | Very high for generative MILP and learning-to-optimize |
