# Deep Notes

## Label Construction
- Paper-stated facts: Tight constraints at the optimal solution are treated as candidate labels, and a subset of critical tight constraints is selected by an information-theory-guided heuristic.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need to verify the exact heuristic definition and whether labels depend on solver traces or only the optimal solution.

## Output Semantics
- Paper-stated facts: The model predicts which inequality constraints should be reduced to equalities in the simplified MILP.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need to verify whether outputs are binary scores, rankings, or calibrated probabilities.

## Loss and Optimization Target
- Paper-stated facts: The paper is clearly framed as a supervised prediction task over critical constraints, but the exact objective is not explicit in the quick scan.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Confirm the exact loss, class imbalance handling, and whether there is any auxiliary objective.

## Inference and Search
- Paper-stated facts: Predicted constraints are fixed as equalities and the reduced MILP is passed to a standard solver.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need to verify how aggressive reduction is selected and how infeasibility is controlled in practice.

## Real Novelty
- Paper-stated facts: The paper argues that most reduction work focuses on variables, while this work centers constraint reduction and uses abstract-level information.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need a deeper read to compare it carefully against recent variable-reduction and predict-and-search MILP papers.

## Ablation Support
- Paper-stated facts: The abstract claims strong improvements in solution quality and runtime against state-of-the-art MILP solvers.
- Evidence: Abstract
- Open questions: Need the full experimental section to see whether the gains come from the representation, the heuristic, or both.

## Weaknesses / Ambiguities
- Paper-stated facts: Many implementation details remain abstract-level in this first pass, especially benchmark composition and training loss.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need to verify solver compatibility details, failure cases, and whether benefits hold across diverse MILP families.

## My Interpretation
- Inference: This is a strong paper if your interest is MILP-specific learning-to-optimize beyond branching, especially model reduction and solver acceleration.
- Confidence: High on the headline idea, medium on method internals.
- What to verify next: Exact benchmark domains, objective function, and how the multi-modal encoder is instantiated.
