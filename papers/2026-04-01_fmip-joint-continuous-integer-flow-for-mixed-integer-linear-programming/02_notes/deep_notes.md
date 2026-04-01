# Deep Notes

## Label Construction
- Paper-stated facts: The paper learns a distribution over full MILP solutions rather than a single-step deterministic prediction, with both integer and continuous variables included.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need to confirm the exact data generation process and whether training samples are optimal solutions only or a broader solution set.

## Output Semantics
- Paper-stated facts: The model outputs a complete candidate MILP solution with both discrete and continuous components.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need to verify whether outputs are produced in one shot at the final step or through intermediate denoising/flow states only.

## Loss and Optimization Target
- Paper-stated facts: FMIP explicitly uses conditional flow matching as its generative training paradigm.
- Evidence: Section 1 Introduction
- Open questions: Need the exact loss equations, parameterization of flows, and any auxiliary regularizers.

## Inference and Search
- Paper-stated facts: A holistic guidance mechanism uses complete objective and feasibility feedback during sampling to improve generated solutions.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need to verify how guidance strength is tuned and how it affects runtime.

## Real Novelty
- Paper-stated facts: Prior generative MILP methods model only integer variables; FMIP's claimed novelty is full joint continuous-integer modeling.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need to compare directly against the strongest integer-only diffusion and flow baselines.

## Ablation Support
- Paper-stated facts: The paper reports improvements on eight standard MILP benchmarks and emphasizes compatibility with different backbones and solvers.
- Evidence: Abstract
- Open questions: Need a full read to see whether gains mainly come from joint modeling, guidance, or both.

## Weaknesses / Ambiguities
- Paper-stated facts: The abstract-level scan does not reveal all solver-integration details or the exact benchmark mix.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need to inspect failure cases, scaling behavior, and feasibility guarantees.

## My Interpretation
- Inference: This is the strongest of the three if you care about 2026-style generative MILP heuristics rather than branching or reduction alone.
- Confidence: High on the core idea, medium on implementation specifics.
- What to verify next: Exact loss equations, benchmark families, and solver plug-in protocol.
