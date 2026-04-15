# Slide Outline

## Slide 1
### Title
Apollo-MILP: What Problem Is It Solving?
### Bullets
- ML for MILP solution prediction often fixes predicted variable values to shrink the problem.
- If those predictions are wrong, direct fixing can create poor reduced problems or infeasibility.
- Apollo-MILP asks a stricter question: which predicted values are reliable enough to fix safely?
### Suggested Figure
- Figure 1
### Speaker Notes
- Frame the paper as a response to the fixing-versus-search trade-off in Neural Diving and Predict-and-Search.
- Emphasize that the target is not only better prediction, but safer problem reduction.

## Slide 2
### Title
From ND and PS to Apollo-MILP
### Bullets
- ND aggressively reduces dimension by fixing variables, but can be brittle when predictions are inaccurate.
- PS improves feasibility via trust-region search, but keeps a larger search space and reduces less aggressively.
- Apollo-MILP tries to keep the reduction strength of fixing while borrowing corrective feedback from search.
### Suggested Figure
- Figure 1
### Speaker Notes
- Use this slide to establish the exact baseline gap the paper claims to close.
- The paper's core move is "predict, then verify before fixing," not "predict harder."

## Slide 3
### Title
Core Framework: Alternating Prediction and Correction
### Bullets
- Each round predicts a partial solution on the current original or reduced MILP.
- A trust-region search produces a reference solution that corrects the predictor.
- Only reliable variables are fixed, yielding the next reduced MILP.
- The process repeats until the final round returns the best solver solution.
### Suggested Figure
- Figure 2
### Speaker Notes
- Walk through the loop in Algorithm 1.
- The important systems-level insight is that solver feedback is used inside the reduction loop, not only at the end.

## Slide 4
### Title
Prediction Step: Graph Encoder, Targets, and Loss
### Bullets
- The predictor is a bipartite-graph GNN over constraints and variables.
- It outputs binary marginals $p_\theta(x_i = 1 \mid I)$ for the current MILP.
- Training targets are energy-weighted marginals computed from pools of optimal or near-optimal solutions.
- Training minimizes cross-entropy in Eq. (4), with data augmentation over reduced instances.
### Suggested Figure
- Section 4.1 equations and Table 6
### Speaker Notes
- Stress that the labels are soft marginals, not one-hot assignments.
- The data augmentation step matters because the model must operate on reduced problems at test time.

## Slide 5
### Title
Correction Step: Trust-Region Search and UEBO
### Bullets
- Trust-region search produces a reference solution $\tilde{x}$ near the predicted partial solution.
- UEBO upper-bounds the mismatch between predictor marginals and the unknown optimal-solution distribution.
- In practice, the paper uses prediction-correction consistency as a simple fixing rule.
- Variables are fixed only when prediction and correction agree: $P' = \\{ i \\in P \\mid \\hat{x}_i = \\tilde{x}_i \\}$.
### Suggested Figure
- Figure 2; Eq. (5); Eq. (7)
### Speaker Notes
- Clarify that UEBO mixes predictor uncertainty and predictor-reference discrepancy.
- The practical decision rule is simpler than the full metric, which is important for implementation.

## Slide 6
### Title
Theory: Why the Fixing Rule Should Help
### Bullets
- Theorem 2 links lower UEBO to higher prediction-correction consistency.
- Theorem 3 states that consistent variables have higher precision than using prediction-only or reference-only values.
- Corollary 4 gives a feasibility guarantee relative to the trust-region search problem.
### Suggested Figure
- Theorem 2; Theorem 3; Corollary 4
### Speaker Notes
- Position the theory as justification for "safe fixing," not as a proof of global optimality.
- This is the paper's main argument for why correction should improve reduction quality.

## Slide 7
### Title
Experiments: Benchmarks and Main Results
### Bullets
- Main evaluation uses CA, SC, IP, and WA with 240/60/100 train-validation-test splits.
- Apollo-MILP outperforms ND, PS, and ConPS across the reported benchmarks.
- The paper reports over 80% absolute-gap reduction versus Gurobi and over 30% versus SCIP on average.
- On IP and WA, Apollo-MILP finds better solutions in 1,000 seconds than Gurobi reaches in 3,600 seconds.
### Suggested Figure
- Table 1; Figure 3; Table 7
### Speaker Notes
- Separate "better gap" from "better objective orientation" because CA is maximization while the others are minimization.
- Highlight that the paper claims both quality gains and strong reduction efficiency.

## Slide 8
### Title
Takeaways, Real Novelty, and What to Verify Next
### Bullets
- The main novelty is a solver-in-the-loop prediction-correction reduction framework for MILP.
- UEBO and the consistency rule operationalize "fix only what correction agrees with."
- Strong ablations support the fixing strategy, warm-start comparison, and data augmentation.
- Remaining reading tasks: appendix implementation details, general-MILP extension details, and solver-sensitivity analysis.
### Suggested Figure
- Table 2; Table 17
### Speaker Notes
- End by distinguishing the paper from work that only upgrades the predictor.
- This slide is also a good bridge into deeper reading or a presentation Q&A.
