# Paper Brief

## Paper Metadata
- Title: Apollo-MILP: An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear Programming
- Authors: Haoyang Liu, Jie Wang, Zijie Geng, Xijun Li, Yuxuan Zong, Fangzhou Zhu, JianYe Hao, Feng Wu
- Venue / Year: ICLR 2025
- Paper Folder: papers/2026-04-13_apollo-milp-an-alternating-prediction-correction-neural-solving-framework-for-mixed-integer-linear-programming

## 1. Research Question
- Summary: Can ML-based MILP solution prediction be made reliable enough for aggressive problem reduction by alternating neural prediction with solver-based correction, and then fixing only the variable values that appear trustworthy?
- Evidence: Abstract; Section 1 Introduction; Figure 1

## 2. High-Level Algorithmic Idea
- Summary: Apollo-MILP iteratively alternates a GNN prediction step and a trust-region correction step. It compares the predicted partial solution with a solver-produced reference solution, uses UEBO and its consistency approximation to identify reliable variables, fixes only those variables, and repeats on the reduced MILP.
- Evidence: Abstract; Section 4.1; Section 4.2; Section 4.3; Figure 2; Algorithm 1

## 3. Training Label / Dataset
- Summary: The predictor is trained on energy-weighted marginal targets derived from pools of optimal or near-optimal solutions for both original and augmented reduced instances. Main experiments use the CA, SC, IP, and WA benchmarks with 240 training, 60 validation, and 100 test instances.
- Evidence: Section 4.1; Eq. (3); Eq. (4); Section 5.1; Table 7; Appendix D

## 4. Network Structure
- Summary: The learning component is a bipartite-graph GNN predictor over constraint and variable nodes. The implementation follows the PS-style predictor and uses four half-convolution layers to output binary-variable marginals for the current original or reduced MILP.
- Evidence: Section 3.2; Section 4.1; Appendix D; Table 6

## 5. Network Output
- Summary: The model outputs marginal probabilities $p_\theta(x_i = 1 \mid I)$ for binary variables. These marginals are converted into a partial solution by selecting the $k_1$ highest-probability variables to fix to 1 and the $k_0$ lowest-probability variables to fix to 0.
- Evidence: Section 3.3; Section 4.1; Algorithm 1

## 6. Loss Function
- Summary: The predictor is trained with cross-entropy between predicted binary marginals and the energy-weighted target marginals computed from a solution pool for each instance. UEBO is used at correction time as a reliability metric, not as the supervised training loss.
- Evidence: Section 4.1; Eq. (3); Eq. (4); Section 4.2; Eq. (5)

## 7. Inference Policy
- Summary: At inference time Apollo-MILP runs alternating rounds of prediction and correction: predict a partial solution, run trust-region search to obtain a reference solution, keep only prediction-correction consistent variables, fix them to reduce the MILP, and return the best solver solution from the final round. The experiments instantiate this with four rounds and a 100/100/200/600 second budget split.
- Evidence: Section 4.2; Section 4.3; Eq. (7); Algorithm 1; Section 5.1

## 8. Novel Contribution
- Summary: The paper introduces the first alternating prediction-correction neural solving framework for MILP solution prediction, proposes UEBO plus a simple consistency-based fixing rule with precision and feasibility guarantees, and demonstrates stronger reduction quality than ND, PS, and ConPS on standard benchmarks.
- Evidence: Abstract; Section 1 Introduction; Eq. (5); Eq. (7); Theorem 3; Corollary 4; Table 1; Table 2

## Evidence Pointers
- Abstract: Apollo-MILP overview, UEBO motivation, and headline empirical gains
- Figure 1: ND and PS baseline behaviors and why fixing versus search matters
- Figure 2: end-to-end Apollo-MILP workflow
- Section 4.1; Eq. (3); Eq. (4): target construction, data augmentation, and supervised loss
- Section 4.2; Eq. (5): UEBO definition and correction logic
- Section 4.3; Eq. (7); Theorem 3; Corollary 4; Algorithm 1: consistency rule, guarantees, and inference loop
- Section 5.1; Table 7; Appendix D: benchmarks, splits, implementation details
- Table 1; Table 2: main results and fixing-strategy ablation

## Unresolved Questions
- The paper states that mixed-binary simplifications can be extended to general MILP using prior techniques, but the exact extension details are not unpacked in this pass.
- The method depends on solver-produced reference solutions; the paper does not fully isolate how sensitive UEBO-style fixing is to correction quality from different solvers or search budgets.
