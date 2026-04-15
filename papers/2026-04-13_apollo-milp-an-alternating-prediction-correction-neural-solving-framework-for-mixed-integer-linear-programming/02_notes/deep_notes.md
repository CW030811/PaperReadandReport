# Deep Notes

## Label Construction
- Paper-stated facts: Apollo-MILP does not train on a single incumbent label. It builds energy-weighted marginal targets from pools of optimal or near-optimal solutions, and augments training with reduced instances created by fixing random subsets of variables to values from a strong solution.
- Evidence: Section 4.1; Eq. (3); Eq. (4); Appendix D
- Open questions: How sensitive the learned marginals are to the quality and diversity of the collected solution pool is not fully analyzed.

## Output Semantics
- Paper-stated facts: The predictor outputs $p_\theta(x_i = 1 \mid I)$ for binary variables under an independence assumption. A partial solution is formed by taking the highest and lowest marginal values, and the correction stage further filters that partial solution down to prediction-correction consistent variables.
- Evidence: Section 3.3; Section 4.1; Eq. (7); Algorithm 1
- Open questions: The paper points to prior work for extending the mixed-binary setup to broader MILPs, but that mapping is not expanded in the main text.

## Loss and Optimization Target
- Paper-stated facts: The supervised loss is the cross-entropy between predicted marginals and energy-weighted target marginals from the solution pool. UEBO is introduced later as an upper bound on prediction error relative to the optimal solution distribution, but it is used for correction-time reliability assessment rather than direct training.
- Evidence: Section 4.1; Eq. (3); Eq. (4); Section 4.2; Eq. (5)
- Open questions: The paper does not directly optimize final primal gap or solver runtime end-to-end; it relies on the two-stage interaction between learned prediction and solver correction.

## Inference and Search
- Paper-stated facts: Apollo-MILP alternates prediction and correction across multiple rounds. In each round it predicts a partial solution, solves a trust-region search problem to get a reference solution, keeps only consistent variable assignments, and fixes them in the next reduced problem. The experimental setup uses four rounds with a 100/100/200/600 second split.
- Evidence: Section 4.2; Section 4.3; Algorithm 1; Section 5.1
- Open questions: Runtime and reduction quality likely depend on solver strength and benchmark-specific trust-region hyperparameters more than the paper isolates.

## Real Novelty
- Paper-stated facts: The paper's core novelty is not just a stronger predictor. It introduces a solver-in-the-loop correction mechanism, UEBO as an uncertainty-and-disagreement upper bound, and a simple consistency rule with theoretical precision and feasibility guarantees for safe variable fixing.
- Evidence: Abstract; Section 1 Introduction; Eq. (5); Eq. (7); Theorem 3; Corollary 4
- Open questions: The exact incremental contribution of UEBO as a concept versus the practical consistency rule could be unpacked more carefully in a slower second read.

## Ablation Support
- Paper-stated facts: The ablation section compares consistency-based fixing against direct prediction fixing and direct reference fixing, and also compares the framework against warm-starting alternatives. Appendix experiments further show gains from data augmentation and from choosing the right number of iteration rounds.
- Evidence: Section 5.3; Table 2; Table 3; Table 16; Table 17
- Open questions: A fuller appendix pass would help quantify how robust each ablation trend is across all auxiliary benchmarks, not just IP and WA.

## Weaknesses / Ambiguities
- Paper-stated facts: The main formulation depends on binary-variable marginals, an independence assumption, and a solver-based correction stage. In practice the paper also replaces the full UEBO computation with a consistency-based fixing rule, so the deployed policy is simpler than the theoretical metric.
- Evidence: Section 3.3; Section 4.2; Section 4.3; Appendix discussion around UEBO approximation
- Open questions: It remains unclear how well the framework would work when the trust-region search produces weak reference solutions or when solver calls are expensive enough to dominate total runtime.

## My Interpretation
- Inference: Apollo-MILP is best understood as a safe reduction framework rather than a pure predictor. The predictor proposes values, but the solver-backed correction stage decides which values are trustworthy enough to freeze.
- Confidence: medium-high
- What to verify next: Read Appendix D and Appendix H more slowly to extract the exact predictor stack, all hyperparameter choices, and the stronger generalization/real-world claims.
