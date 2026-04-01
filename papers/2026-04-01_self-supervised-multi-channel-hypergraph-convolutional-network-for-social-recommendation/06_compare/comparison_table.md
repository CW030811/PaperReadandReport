# Comparison Table

| Paper | Task | Label Type | Dataset | Encoder | Decoder | Output | Loss | Inference | Novelty | Weakness | Relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Yu et al. (WWW 2021) | Social recommendation with high-order user relations | Observed user-item interactions plus self-supervised hypergraph structure | Multiple real-world social recommendation datasets; exact names uncertain in quick scan | Multi-channel hypergraph convolutional network | Recommendation scoring head over user-item embeddings | Recommendation ranking scores | Recommendation objective plus mutual-information self-supervision | Rank candidate items for each user | Captures high-order social relations with motif-induced hypergraphs | Not a strict discrete choice paper | Medium-high as a backup paper for social influence and graph-based choice behavior |
