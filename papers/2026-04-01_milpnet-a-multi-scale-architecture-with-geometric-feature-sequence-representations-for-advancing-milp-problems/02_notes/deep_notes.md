# Deep Notes

## Label Construction
- Paper-stated facts: The abstract frames the model as learning mappings related to feasibility, objective value, and optimal solutions rather than only one prediction task.
- Evidence: OpenReview abstract
- Open questions: Need to verify exactly which supervision targets are used in training and whether they are learned jointly.

## Output Semantics
- Paper-stated facts: Outputs can approximate feasibility, optimal objective value, and optimal solution mappings.
- Evidence: OpenReview abstract
- Open questions: Need to verify how these multiple outputs are parameterized and evaluated.

## Loss and Optimization Target
- Paper-stated facts: The abstract-level scan does not expose the exact loss function.
- Evidence: OpenReview abstract
- Open questions: Need to inspect whether the model uses one shared objective or multiple task-specific losses.

## Inference and Search
- Paper-stated facts: MILPnet is used as a learned representation within an end-to-end solver pipeline on real-world MILP benchmarks.
- Evidence: OpenReview abstract
- Open questions: Need to see where exactly it plugs into the solver and whether it replaces or augments standard components.

## Real Novelty
- Paper-stated facts: The paper directly challenges GNN expressiveness for MILP, especially under Foldable instances constrained by Weisfeiler-Lehman-style limits.
- Evidence: OpenReview abstract
- Open questions: Need to inspect the theoretical argument and how broadly the Foldable-instance critique applies in practice.

## Ablation Support
- Paper-stated facts: The abstract claims better feasibility prediction, faster convergence, fewer parameters, scale generalization, and strong real-world benchmark performance.
- Evidence: OpenReview abstract
- Open questions: Need to inspect which gains come from sequence representation, attention scale design, or theory-driven construction.

## Weaknesses / Ambiguities
- Paper-stated facts: The quick scan gives a strong high-level claim but leaves many implementation and evaluation details unspecified.
- Evidence: OpenReview abstract
- Open questions: Need exact losses, benchmark list, and solver integration details before strong conclusions.

## My Interpretation
- Inference: This is the best paper of the three if your current question is whether GNNs are even the right representation class for MILP.
- Confidence: High on the paper's positioning, medium on method details.
- What to verify next: Foldable MILP definition, exact attention architecture, and what solver tasks the model supervises.
