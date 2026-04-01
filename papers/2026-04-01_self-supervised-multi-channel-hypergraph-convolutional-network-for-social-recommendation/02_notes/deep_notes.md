# Deep Notes

## Label Construction
- Paper-stated facts: The recommendation task is supervised by observed user-item interactions, while self-supervision is derived from hypergraph structure.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need to verify exact training split and whether negative sampling is used.

## Output Semantics
- Paper-stated facts: The model learns user and item embeddings for recommendation ranking.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need to confirm exact scoring function and ranking objective.

## Loss and Optimization Target
- Paper-stated facts: The paper adds a self-supervised task based on hierarchical mutual information maximization between user, sub-hypergraph, and global hypergraph representations.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need to confirm the exact primary recommendation loss and how it is balanced with the auxiliary task.

## Inference and Search
- Paper-stated facts: At inference time, learned embeddings are used to produce recommendation rankings for users.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need to verify whether all hypergraph channels are used symmetrically at test time.

## Real Novelty
- Paper-stated facts: The main novelty is exploiting motif-induced hypergraphs to capture high-order user relation patterns that pairwise social GNNs miss.
- Evidence: Abstract; Section 1 Introduction; Figure 1
- Open questions: Need to compare more carefully against other high-order graph or hypergraph recommender models from the same period.

## Ablation Support
- Paper-stated facts: The abstract states that ablation studies validate both the multi-channel design and the self-supervised task.
- Evidence: Abstract
- Open questions: Need the experiments section to see which component contributes most.

## Weaknesses / Ambiguities
- Paper-stated facts: This is a social recommendation paper, not a strict discrete choice paper, so its relevance is conceptual rather than task-identical.
- Evidence: Abstract; Section 1 Introduction
- Open questions: Need dataset details and exact training objective before using it as a very close methodological analogue.

## My Interpretation
- Inference: This is a good backup read if you care about social influence, high-order relations, and graph-based user-choice modeling.
- Confidence: High on the paper's positioning and core method, medium on training details.
- What to verify next: Exact datasets, scoring loss, and whether the multi-channel hypergraph idea transfers to your setting.
