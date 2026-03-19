# Deep Notes

## Label Construction
- Paper-stated facts: The supervision signal is not a full solution, node value, or final tree-size label. It is a dataset of state-action pairs collected during branch-and-bound, where each label is the strong branching expert's chosen branching variable at a node.
- Paper-stated facts: The authors explicitly frame the task as behavioral cloning from expert decisions, not score regression or ranking prediction. This matters because the label is a discrete action, not a scalar score.
- Paper-stated facts: When collecting data, they had to re-implement a side-effect-free expert called `vanillafullstrong`, because SCIP's built-in strong branching changes solver state and therefore is not a clean oracle for dataset extraction.
- Paper-stated facts: The training corpus is benchmark-specific: for each of the four MILP families they collect 100,000 training samples and 20,000 validation samples by sampling instances and recording node states plus expert actions during SCIP runs.
- Evidence: Section 4.1; Supplementary Section 1 Dataset collection details; Supplementary Section 2.1
- Open questions: The paper does not quantify how much oracle noise or tie ambiguity exists when multiple candidates are similarly good under strong branching.

## Output Semantics
- Paper-stated facts: The network output is a masked softmax distribution over candidate branching variables, meaning currently eligible non-fixed LP variables at the focused B&B node.
- Paper-stated facts: This output should not be interpreted as a calibrated estimate of objective gain, subtree size, or true strong branching score. It is a classifier over actions.
- Paper-stated facts: The model discards constraint nodes at the end and produces scores only on variable nodes, so the network is directly optimized to choose "which variable to branch on next."
- Evidence: Section 4.3; Figure 2
- Open questions: The paper does not study probability calibration, only ranking/selection accuracy such as acc@1, acc@5, and acc@10.

## Loss and Optimization Target
- Paper-stated facts: The optimized loss is cross-entropy over expert actions, i.e. negative log-likelihood of the strong branching decision.
- Paper-stated facts: This is a surrogate objective, not the final metric of interest. The real downstream goals are lower solve time and fewer B&B nodes, but the model is not trained directly on those quantities.
- Paper-stated facts: The authors deliberately avoid reinforcement learning because episode length depends on policy quality, early policies are too poor, and the induced MDPs are very large once an instance is fixed.
- Paper-stated facts: They pretrain prenorm layers, then optimize with Adam using minibatch size 32 and initial learning rate 1e-3, decaying on validation plateaus.
- Evidence: Section 4.1 Equation (3); Section 4 Methodology; Supplementary Section 2.1
- Open questions: The paper does not analyze whether a better imitation accuracy always translates into better solving time; it only shows the correlation empirically in the reported experiments.

## Inference and Search
- Paper-stated facts: The learned GCNN replaces only the variable selection rule inside branch-and-bound. It does not replace the solver, LP relaxations, node selection, primal heuristics, cuts, or the rest of SCIP's search machinery.
- Paper-stated facts: Evaluation is done inside SCIP 6.0.1 with cutting planes allowed only at the root node and restarts deactivated; all other solver parameters remain at default values.
- Paper-stated facts: Strong branching itself is highlighted as an example of high decision quality but poor wall-clock speed. The learned policy is meant to approximate that decision quality at much lower per-node inference cost.
- Paper-stated facts: The authors explicitly discuss an inference-speed versus policy-capacity tradeoff: deeper or larger GCNNs reduced node counts slightly but worsened total solve time due to slower per-decision inference.
- Evidence: Section 3.3; Section 5.1; Section 6 Discussion
- Open questions: The paper does not isolate how much of the final gain comes from pure branch quality versus interactions with the remaining solver components.

## Real Novelty
- Paper-stated facts: The paper is not the first to imitate strong branching. Prior work already used imitation learning for branching, often with ranking or regression formulations and hand-engineered features.
- Paper-stated facts: The actual novelty is the combination of a bipartite MILP graph state representation, GCNN policy architecture, and classification-style imitation of expert decisions rather than prediction of expert scores or rankings.
- Paper-stated facts: A second meaningful novelty claim is experimental: they compare against a much more realistic full solver setup and emphasize generalization to larger instances than seen during training.
- Evidence: Section 1; Section 2 Related work; Section 5.2; Section 7
- Open questions: The paper does not fully disentangle how much improvement comes from the graph representation itself versus the switch from ranking/regression to action classification.

## Ablation Support
- Paper-stated facts: The ablation study tests three convolution variants on set covering only: mean aggregation, sum aggregation without prenorm, and sum aggregation with prenorm.
- Paper-stated facts: The authors use this to support two architectural claims: sum convolutions are a better prior than mean convolutions for branching, and prenorm improves stability and generalization on larger instances.
- Paper-stated facts: On small instances the variants are similar, while on larger instances the proposed sum-plus-prenorm model performs better.
- Evidence: Section 5.3; Table 3
- Open questions: The ablation is narrow because it is restricted to one benchmark family and does not vary many other design choices such as number of message-passing layers or feature subsets.

## Weaknesses / Ambiguities
- Paper-stated facts: The state representation is only a subset of the full solver state, which the authors acknowledge makes the problem technically a partially observable MDP.
- Paper-stated facts: Feature details are not fully specified in the main text and partly pushed to supplementary material and source code, which makes exact reproduction harder from the paper alone.
- Paper-stated facts: Generalization is strong but not unlimited. The authors explicitly say performance drops as test instances get much larger, and maximum independent set is particularly challenging.
- Paper-stated facts: The training target is expert imitation, not direct optimization of solve time or node count, so there is an unavoidable objective mismatch between what is optimized and what is ultimately evaluated.
- Evidence: Section 4.2; Section 5.2; Section 6 Discussion; Supplementary Section 2.1
- Open questions: It remains unclear how robust the method is when solver settings, cut behavior, or benchmark distributions shift substantially away from the training setup.

## My Interpretation
- Inference: For your usual reading dimensions, this paper is unusually clean on the `output / loss / inference` chain. The network outputs an action distribution over branch candidates, the loss trains it to imitate the expert's chosen action, and inference uses that same action space inside SCIP. So local alignment is good.
- Inference: The bigger mismatch is one level higher: the paper trains for expert imitation but evaluates on global solve efficiency. That mismatch is acceptable here because strong branching is a high-quality oracle, but it is still a surrogate-learning setup rather than end-to-end optimization.
- Inference: The final solver behavior still depends on classical search outside the network. So this is best understood as "learned branching inside a traditional exact solver," not "a neural combinatorial optimizer that directly solves MILPs."
- Confidence: High
- What to verify next: If you want a still deeper read, the next worthwhile pass is to inspect the supplementary feature table and compare this paper's action-classification setup against later learning-to-branch papers that optimize stronger downstream objectives or use reinforcement learning.
