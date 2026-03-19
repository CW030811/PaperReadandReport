# Paper Brief

## Paper Metadata
- Title: Exact Combinatorial Optimization with Graph Convolutional Neural Networks
- Authors: Maxime Gasse, Didier Chetelat, Nicola Ferroni, Laurent Charlin, Andrea Lodi
- Venue / Year: NeurIPS 2019
- Paper Folder: papers/2026-03-19_exact-combinatorial-optimization-with-graph-convolutional-neural-networks

## 1. Research Question
- Summary: Can a graph convolutional neural network learn a branch-and-bound variable selection policy for MILPs that approximates strong branching, reduces manual feature engineering, and generalizes to larger instances than seen during training?
- Evidence: Abstract; Section 1 Introduction; Section 4 Methodology

## 2. High-Level Algorithmic Idea
- Summary: Represent each B&B decision state as a variable-constraint bipartite graph of the current MILP LP relaxation, encode that state with a GCNN policy, and train the policy by behavioral cloning from strong branching decisions instead of predicting scores or rankings directly.
- Evidence: Abstract; Section 1; Section 4.1; Section 4.2; Figure 2

## 3. Training Label / Dataset
- Summary: Labels are expert actions from strong branching. For each benchmark family, the authors generate random MILP instances, solve them with SCIP, and record state-action pairs from B&B nodes. They report 100,000 training samples, 20,000 validation samples, and a matching test sampling protocol per benchmark, across set covering, combinatorial auction, capacitated facility location, and maximum independent set.
- Evidence: Section 4.1; Section 5.1 Training; Supplementary Section 1 Dataset collection details

## 4. Network Structure
- Summary: The model is a bipartite GCNN over constraint nodes, variable nodes, and sparse edge features. It applies one graph convolution as two interleaved half-convolutions, variable-to-constraint then constraint-to-variable, using 2-layer MLPs for message/update functions. A final 2-layer perceptron on variable nodes outputs branch scores, with sum aggregation and prenorm layers used in the main model.
- Evidence: Figure 2; Section 4.2; Section 4.3; Section 5.3 Ablation study

## 5. Network Output
- Summary: The network outputs a masked softmax probability distribution over candidate branching variables, meaning the currently fractional, non-fixed variables at the focused B&B node.
- Evidence: Section 4.3 Policy parametrization; Figure 2

## 6. Loss Function
- Summary: The training objective is behavioral cloning with cross-entropy loss over expert decisions, minimizing negative log-probability of the strong-branching-selected action. Training uses Adam after prenorm pretraining.
- Evidence: Section 4.1 Equation (3); Supplementary Section 2.1 Training details

## 7. Inference Policy
- Summary: At each B&B step, the solver queries the GCNN policy on the current node state, ranks candidate branching variables via the masked softmax scores, and branches on the top predicted candidate as the learned variable selection rule inside SCIP.
- Evidence: Section 3.3 MDP formulation; Section 4.3; Section 5.1 Evaluation and accuracy definition

## 8. Novel Contribution
- Summary: The paper's main novelty is framing MILP branching as imitation learning over a bipartite GCNN state representation, replacing heavy feature engineering with graph message passing and directly cloning expert branching decisions. It also demonstrates strong generalization to larger problem instances and favorable comparison against a full solver's default branching rule.
- Evidence: Abstract; Section 1; Section 2 Related work; Section 5.2 Comparative experiment; Section 7 Conclusion

## Evidence Pointers
- Section 1: problem framing, contributions, and benchmark families
- Section 4.1: imitation learning setup and cross-entropy objective
- Section 4.2: bipartite state encoding for MILP branch-and-bound states
- Section 4.3: GCNN architecture, masked softmax output, and prenorm layers
- Figure 2: state representation and policy architecture
- Equation (3): behavioral cloning cross-entropy objective
- Section 5.1: benchmark setup and training/evaluation protocols
- Section 5.2: comparative results and claims about generalization and solver performance
- Supplementary Section 1: dataset collection details
- Supplementary Section 2.1: optimizer and training hyperparameters

## Unresolved Questions
- The main text does not fully enumerate every input feature dimension inline; some exact feature details are deferred to the supplementary materials and code.
- The inference-time branch choice is functionally the top-ranked masked-softmax candidate, but the paper emphasizes ranking/probability more than a separate decoding rule.
