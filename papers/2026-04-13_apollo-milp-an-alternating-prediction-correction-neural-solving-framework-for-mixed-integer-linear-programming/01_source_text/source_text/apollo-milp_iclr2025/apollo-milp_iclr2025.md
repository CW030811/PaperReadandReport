# APOLLO-MILP: AN ALTERNATING PREDICTION-CORRECTION NEURAL SOLVING FRAMEWORK FOR MIXED-INTEGER LINEAR PROGRAMMING

Haoyang Liu<sup>1</sup><sup>∗</sup> , Jie Wang<sup>1</sup>† , Zijie Geng<sup>1</sup> , Xijun Li4,2, Yuxuan Zong<sup>1</sup> , Fangzhou Zhu<sup>2</sup> , JianYe Hao2,3 , Feng Wu<sup>1</sup>

<sup>1</sup>MoE Key Laboratory of Brain-inspired Intelligent Perception and Cognition, University of Science and Technology of China

- <sup>2</sup> Noah's Ark Lab, Huawei Technologies
- <sup>3</sup> Tianjin University
- <sup>4</sup> Shanghai Jiao Tong University

### ABSTRACT

# Smart Initial Basis Selection

Leveraging machine learning (ML) to predict an initial solution for mixed-integer linear programming (MILP) has gained considerable popularity in recent years. These methods predict a solution and fix a subset of variables to reduce the problem dimension. Then, they solve the reduced problem to obtain the final solutions. However, directly fixing variable values can lead to low-quality solutions or even infeasible reduced problems if the predicted solution is not accurate enough. To address this challenge, we propose an Alternating prediction-correction neural solving framework (Apollo-MILP) that can identify and select accurate and reliable predicted values to fix. In each iteration, Apollo-MILP conducts a prediction step for the unfixed variables, followed by a correction step to obtain an improved solution (called reference solution) through a trust-region search. By incorporating the predicted and reference solutions, we introduce a novel Uncertainty-based Error upper BOund (UEBO) to evaluate the uncertainty of the predicted values and fix those with high confidence. A notable feature of Apollo-MILP is the superior ability for problem reduction while preserving optimality, leading to high-quality final solutions. Experiments on commonly used benchmarks demonstrate that our proposed Apollo-MILP significantly outperforms other ML-based approaches in terms of solution quality, achieving over a 50% reduction in the solution gap.

### 1 INTRODUCTION

Mixed-integer linear programming (MILP) is one of the most fundamental models for combinatorial optimization with broad applications in operations research [\(Bixby et al., 2004\)](#page-10-0), engineering [\(Ma](#page-13-0) [et al., 2019\)](#page-13-0), and daily scheduling or planning [\(Li et al., 2024b\)](#page-13-1). However, solving large-size MILPs remains time-consuming and computationally expensive, as many are NP-hard and have exponential expansion of search spaces as instance sizes grow. To mitigate this challenge, researchers have explored a wide suite of machine learning (ML) methods [\(Gasse et al., 2022\)](#page-11-0). In practice, MILP instances from the same scenario often share similar patterns and structures, which ML models can capture to achieve improved performance [\(Bengio et al., 2021\)](#page-10-1).

Recently, extensive research has focused on using ML models to predict solutions for MILPs. Notable approaches include Neural Diving (ND) [\(Nair et al., 2020;](#page-14-0) [Yoon, 2021;](#page-16-0) [Paulus & Krause, 2023\)](#page-14-1) and Predict-and-Search (PS) [\(Han et al., 2023;](#page-12-0) [Huang et al., 2024\)](#page-12-1), as illustrated in Figure [1.](#page-1-0) Given a MILP instance, ND and PS begin by employing an ML model to predict an initial solution. ND with SelectiveNet [\(Nair et al., 2020\)](#page-14-0) assigns fixed values to a subset of variables based on the prediction, thereby constructing a reduced MILP problem with a reduced dimensionality of decision variables. Then, ND solves the reduced problem to obtain the final solutions. However, the fixing strategy

<sup>∗</sup>This work was done when Haoyang Liu was an intern at Huawei.

<sup>†</sup>Corresponding author. Email: jiewangx@ustc.edu.cn.

<span id="page-1-0"></span>Figure 1: Illustration of Neural Diving (ND) and Predict-and-Search (PS). For a given MILP problem, both methods begin by using a GNN predictor to generate an initial solution xˆ and construct a partial solution xˆ[P]. ND then fixes the variable values in this partial solution and optimizes the reduced problem. While PS searches within a neighborhood around the partial solution.

faces several limitations. The solving efficiency and the quality of the final solutions heavily depend on the accuracy of the ML-based predictor for initial solutions [\(Huang et al., 2024\)](#page-12-1), but achieving an accurate ML-based predictor is often challenging due to the complex combinatorial nature of MILPs, insufficient training data, and limited model capacity. Consequently, enforcing variables to fixed values that may not be accurate can misguide the search toward areas that do not contain the optimal solution, leading to low-quality final solutions or even infeasible reduced problems. Instead of fixing variables, PS offers a more effective search strategy for a pre-defined neighborhood of the predicted partial solution, leading to better feasibility and higher-quality final solutions. The trustregion search strategy in PS allows better feasibility but is less effective than the fixing strategy in terms of problem dimension reduction, as it requires a larger search space.

To address the aforementioned challenges, a natural idea is to refine the predicted solutions before fixing them. We observe that the search process in PS provides valuable feedback to enhance solution quality for prediction, an aspect that has been overlooked in existing research. Specifically, the solver guides the searching direction toward the optimal solution while correcting variable values that are inappropriately fixed. Theoretically, incorporating this correction yields higher precision for predicted solutions (please see Theorem [3\)](#page-6-0).

In light of this, we propose a novel MILP optimization approach, called the Alternating Prediction-Correction Neural Solving Framework (Apollo-MILP), that can effectively identify the correct and reliable predicted variable values to fix. In each iteration, Apollo-MILP conducts a prediction step for the unfixed variables, followed by a correction step to obtain an improved solution (called reference solution) through a trust-region search. The reference solution serves as guidance provided by the solver to correct the predicted solution. By incorporating both predicted and reference solutions, we introduce a novel Uncertainty-based Error upper BOund (UEBO) to evaluate the uncertainty of the predicted values and fix those with high confidence. Furthermore, we also propose a straightforward variable fixing strategy based on UEBO. Theoretical results show that this strategy guarantees improved solution quality and feasibility for the reduced problem. Experiments demonstrate that Apollo-MILP reduces the solution gap by over 50% across various popular benchmarks, while also achieving higher-quality solutions in one-third of the runtime compared to traditional solvers.

We highlight our main contributions as follows. (1) A Novel Prediction-Correction MILP Solving Framework. Apollo-MILP is the first framework to incorporate a correction mechanism to enhance the precision of solution predictions, enabling effective problem reduction while preserving optimality. (2) Investigating Effective Problem Reduction Techniques. We rethink the existing problemreduction techniques for MILPs and establish a comprehensive criterion for selecting an appropriate subset of variable values to fix, combining the advantages of existing search and fixing strategies. (3) High Performance across Various Benchmarks. We conduct extensive experiments demonstrating Apollo-MILP's strong performance, generalization ability, and real-world applicability.

### 2 RELATED WORKS

ML-Enhanced Branch-and-Bound Solver In practice, typical MILP solvers, such as SCIP [\(Achterberg, 2009\)](#page-10-2) and Gurobi [\(Gurobi Optimization, 2021\)](#page-12-2), are primarily based on the Branch-and-Bound (B&B) algorithm. ML has been successfully integrated to enhance the solving efficiency of these B&B solvers [\(Bengio et al., 2021;](#page-10-1) [Li et al., 2024a;](#page-13-2) [Gasse et al., 2022;](#page-11-0) [Scavuzzo et al., 2024\)](#page-14-2). Specifically, many researchers have leveraged advanced techniques from imitation and reinforcement learning to improve key heuristic modules. A significant portion of this work aims to learn heuristic policies for selecting variables to branch on [\(Khalil et al., 2016;](#page-12-3) [Gasse et al., 2019;](#page-11-1) [Gupta](#page-11-2) [et al., 2020;](#page-11-2) [Zarpellon et al., 2021;](#page-16-1) [Gupta et al., 2022;](#page-12-4) [Scavuzzo et al., 2022;](#page-14-3) [Lin et al., 2024;](#page-13-3) [Zhang](#page-16-2) [et al., 2024;](#page-16-2) [Kuang et al., 2024\)](#page-12-5), selecting cutting planes [\(Tang et al., 2020;](#page-15-0) [Wang et al., 2023b;](#page-15-1) [2024b;](#page-15-2) [Huang et al., 2022;](#page-12-6) [Balcan et al., 2022;](#page-10-3) [Paulus et al., 2022;](#page-14-4) [Ling et al., 2024;](#page-13-4) [Puigdemont](#page-14-5) [et al., 2024\)](#page-14-5), and determining which nodes to explore next [\(He et al., 2014;](#page-12-7) [Labassi et al., 2022;](#page-12-8) [Liu](#page-13-5) [et al., 2024c\)](#page-13-5). These ML-enhanced methods have demonstrated substantial improvements in solving efficiency. Additionally, extensive research has been dedicated to boosting other critical modules in the B&B algorithm, such as separation [\(Li et al., 2023\)](#page-13-6), scheduling of primal heuristics [\(Khalil et al.,](#page-12-9) [2017;](#page-12-9) [Chmiela et al., 2021\)](#page-11-3), presolving [\(Liu et al., 2024a\)](#page-13-7), data generation [\(Geng et al., 2023;](#page-11-4) [Liu](#page-13-8) [et al., 2023a;](#page-13-8) [2024b\)](#page-13-9) and large neighborhood search [\(Song et al., 2020;](#page-14-6) [Wu et al., 2021;](#page-16-3) [Sonnerat](#page-15-3) [et al., 2021;](#page-15-3) [Huang et al., 2023\)](#page-12-10). Beyond practical applications, theoretical advancements have also emerged to analyze the expressiveness of GNNs for MILPs and LPs [\(Chen et al., 2023a;](#page-11-5)[b\)](#page-11-6), as well as to develop landscape surrogates for ML-based solvers [\(Zharmagambetov et al., 2023\)](#page-16-4).

ML for Solution Prediction Another line of research leverages ML models to directly predict solutions [\(Ding et al., 2020;](#page-11-7) [Yoon, 2021;](#page-16-0) [Khalil et al., 2022;](#page-12-11) [Paulus & Krause, 2023;](#page-14-1) [Zeng et al.,](#page-16-5) [2024;](#page-16-5) [Cai et al., 2024;](#page-11-8) [Li et al., 2025;](#page-13-10) [Geng et al., 2025\)](#page-11-9). Neural Diving (ND) [Nair et al.](#page-14-0) [\(2020\)](#page-14-0) is a pioneering approach in this field. Specifically, ND predicts a partial solution based on coverage rates and utilizes SelectiveNet to determine which predicted variables to fix. To enhance the quality of the final solution, subsequent methods incorporate search mechanisms, such as trust-region search (PS) [Han et al.](#page-12-0) [\(2023\)](#page-12-0); [Huang et al.](#page-12-1) [\(2024\)](#page-12-1) and large neighborhood search [Sonnerat et al.](#page-15-3) [\(2021\)](#page-15-3); [Ye et al.](#page-16-6) [\(2023;](#page-16-6) [2024\)](#page-16-7) with sophisticated neighborhood optimization techniques. In this paper, we focus on ND and PS, both of which have gained significant popularity in recent years.

# 3 PRELIMINARIES

### 3.1 MIXED INTEGER LINEAR PROGRAMMING

A mixed-integer linear programming (MILP) is defined as follows,

$$\min_{\boldsymbol{x} \in \mathbb{R}^n} \quad \boldsymbol{c}^{\top} \boldsymbol{x}, \quad \text{s.t.} \quad \boldsymbol{A} \boldsymbol{x} \leq \boldsymbol{b}, \boldsymbol{l} \leq \boldsymbol{x} \leq \boldsymbol{u}, \boldsymbol{x} \in \mathbb{Z}^p \times \mathbb{R}^{n-p}, \tag{1}$$

where x denotes the n-dimensional decision variables, consisting of p integer components and n−p continuous variables. The vector c ∈ R <sup>n</sup> denotes the coefficients of the objective function, A ∈ R <sup>m</sup>×<sup>n</sup> is the constraint coefficient matrix, and b ∈ R <sup>m</sup> represents the right-hand side terms of the constraints. The vectors l ∈ (R ∪ {−∞}) <sup>n</sup> and u ∈ (R ∪ {+∞}) <sup>n</sup> specify the lower and upper bounds for the variables, respectively. It is reasonable that PS primarily focuses on mixed-binary programming with x ∈ {0, 1} <sup>p</sup> × R n−p for simplification, as it can be easily generalized to general MILPs using the well-established modification techniques proposed in [Nair et al.](#page-14-0) [\(2020\)](#page-14-0).

### <span id="page-2-0"></span>3.2 BIPARTITE GRAPH REPRESENTATION FOR MILPS

A MILP instance can be represented as a weighted bipartite graph G = (W ∪ V, E) [Gasse et al.](#page-11-1) [\(2019\)](#page-11-1), as illustrated in Figure [2.](#page-4-0) In this bipartite graph, the two sets of nodes, W and V, represent the constraints and variables in the MILP instance, respectively. An edge is constructed between a constraint node and a variable node if the variable has a nonzero coefficient in the constraint. For further details on the graph features utilized in this paper, please refer to Appendix [E.](#page-20-0)

#### <span id="page-2-1"></span>3.3 PREDICT-AND-SEARCH

Predict-and-Search (PS) [Han et al.](#page-12-0) [\(2023\)](#page-12-0) is a two-stage MILP optimization framework that utilizes machine learning models to learn the Bernoulli distribution for the solution values of binary variables. It then performs a trust-region search within a neighborhood of the predicted solution xˆ to enhance solution quality. Given a MILP instance I, PS considers approximating the solution

distribution q(x | I) by weighing the solutions with their objective value,

$$q(\boldsymbol{x}\mid\mathcal{I}) = \frac{\exp(-E(\boldsymbol{x},\mathcal{I}))}{\sum_{\boldsymbol{x}'\in\mathcal{S}} \exp(-E(\boldsymbol{x}',\mathcal{I}))}, \text{ where the energy function } E(\boldsymbol{x},\mathcal{I}) = \begin{cases} \boldsymbol{c}^{\top}\boldsymbol{x}, & \text{if } \boldsymbol{x} \text{ is feasible,} \\ +\infty, & \text{otherwise,} \end{cases}$$

and S is a collected set of optimal or near-optimal solutions. PS learns the solution distribution using a GNN model p<sup>θ</sup> and computes the marginal probability pθ(x | I) to predict a solution. To simplify the formulation, PS assumes that the variables are independent, as described in [Nair](#page-14-0) [et al.](#page-14-0) [\(2020\)](#page-14-0), i.e., pθ(x | I) = Q<sup>n</sup> <sup>i</sup>=1 pθ(x<sup>i</sup> | I). PS then selects k<sup>1</sup> binary variables with the highest predicted marginal values pθ(x<sup>i</sup> | I) and fixes them to 1. Similarly, PS fixes k<sup>0</sup> binary variables with the lowest marginal values to 0. The hyperparameters k<sup>0</sup> and k<sup>1</sup> are called partial solution size parameters, and we denote the fixed partial solution as xˆ[P], where P is the index set for the fixed variables with k<sup>0</sup> + k<sup>1</sup> elements. Instead of directly fixing the variables xˆ[P], PS employs a traditional solver, such as SCIP or Gurobi, to explore the neighborhood B<sup>P</sup> (xˆ[P], △) of the predicted partial solution xˆ[P] in search of the best feasible solution. Here △ represents the trust-region radius (neighborhood parameter), and B<sup>P</sup> (xˆ[P], △) = {x[P] ∈ R <sup>n</sup> | ∥xˆ[P] − x[P]∥<sup>1</sup> ≤ △} is the trust region. The neighborhood search process is formulated as the following MILP problem, referred to as the *trust-region search problem*,

<span id="page-3-1"></span>
$$\min_{\boldsymbol{x} \in \mathbb{R}^n} \quad \boldsymbol{c}^\top \boldsymbol{x}, \quad \text{s.t.} \quad \boldsymbol{A} \boldsymbol{x} \leq \boldsymbol{b}, \boldsymbol{l} \leq \boldsymbol{x} \leq \boldsymbol{u}, \\
\boldsymbol{x}[P] \in \mathcal{B}_P(\hat{\boldsymbol{x}}[P], \triangle), \boldsymbol{x} \in \mathbb{Z}^p \times \mathbb{R}^{n-p}. \tag{2}$$

Notice that PS reduces to ND (without SelectiveNet) when the neighborhood parameter △ = 0.

# 4 THE PROPOSED ALTERNATING PREDICTION-CORRECTION FRAMEWORK

"*How to identify and fix a high-quality partial solution*" is a longstanding challenge for ML-based solution prediction approaches. Unlike existing works that primarily focus on enhancing the prediction accuracy of ML models, we offer a new perspective by identifying and selecting the correct and reliable predicted values to fix, thereby improving the quality of final solutions and overall solving efficiency. The proposed Apollo-MILP framework alternates between prediction (Section [4.1\)](#page-3-0) and correction (Section [4.2\)](#page-4-1) steps, progressively identifying high-confidence variables and expanding the subset of fixed variables. As the algorithm proceeds, we obtain a sequence of MILPs I (0) → I(1) → · · · → I(K) with fewer decision variables, where the superscripts {(k) | k = 0, · · · , K} is the iteration number. The overview of our architecture is in Figure [2.](#page-4-0)

As shown in Figure [2,](#page-4-0) during the k th iteration, Apollo-MILP processes a MILP I (k) , which may be either the original problem or a reduced version. In this section, we assume that the prediction and correction steps are performed within the k th iteration. Therefore, we simplify the notation by omitting the superscript (k) without leading to misunderstanding. For example, we denote the predicted solution by xˆ instead of xˆ (k) .

#### <span id="page-3-0"></span>4.1 PREDICTION STEP

In each prediction step, our goal is to predict the solution for the current MILP problem I, which is represented as a bipartite graph, as discussed in Section [3.2.](#page-2-0)

We employ a GNN-based solution predictor p<sup>θ</sup> to predict the marginal probabilities of values pθ(x | I) for binary variables in the optimal solution, similar to the method employed in PS [\(Han et al.,](#page-12-0) [2023\)](#page-12-0). Assuming independence among the variables, the predictor outputs the probability that the variable equals 1, i.e., pθ(x<sup>i</sup> = 1 | I) for i = 1, · · · , n.

As mentioned above, the predictor takes either the original or reduced MILP problems as input. However, the distribution of the reduced problems may differ from that of the original problems. To address this issue, we employ data augmentation to align the distributional shifts. Specifically, for a given MILP instance I in the training dataset, we collect a set S<sup>I</sup> of m optimal or near-optimal solutions to approximate the solution distribution q(x | I) mentioned in Section [3.3.](#page-2-1) We then randomly sample a solution x ∗ from this solution pool SI, along with a subset of variables from I. We fix the selected variables to the corresponding values in x ∗ to generate a reduced instance I ′ .

<span id="page-4-0"></span>Figure 2: The overview of Apollo-MILP. Apollo-MILP operates through an iterative process that alternates between prediction and correction steps to reduce the original MILP problem progressively. In the prediction step, Apollo-MILP (1) employs a GNN to generate a partial solution. In the correction step, (2) a trust region-based search is conducted to refine this solution to obtain the reference solution. (3) The proposed variable fixing criterion, UEBO, is then calculated to identify which variables should be fixed. (4) Finally, we reduce the problem dimension by enforcing the selected variable values to fix values.

For each reduced instance I ′ , we also collect m optimal or near-optimal solutions to estimate the solution distribution q(x | I′ ). All instances and solutions are combined, resulting in an enriched training dataset denoted as D.

To calculate the prediction target for training, we construct the estimated probability target vector (p(x<sup>1</sup> = 1 | I), p(x<sup>2</sup> = 1 | I), · · · , p(x<sup>n</sup> = 1 | I))<sup>⊤</sup> . Here, we let

$$p_i = p(x_i = 1 \mid \mathcal{I}) = \frac{\sum_{x' \in \mathcal{S}_{\mathcal{I}}, x_i' = 1} \exp(-c^{\top} x')}{\sum_{x' \in \mathcal{S}_{\mathcal{I}}} \exp(-c^{\top} x')}$$
(3)

be the probability of variable x<sup>i</sup> being assigned the value 1, given the instance I from the enriched dataset D and the solution set SI. Finally, the predictor p<sup>θ</sup> is trained by minimizing the cross-entropy loss [\(Han et al., 2023\)](#page-12-0)

$$\mathcal{L}(\theta) = -\frac{1}{|\mathcal{D}|} \sum_{(\mathcal{I}, \mathcal{S}_{\mathcal{I}}) \in \mathcal{D}} \sum_{i=1}^{n} (\boldsymbol{p}_{i} \log p_{\theta}(\boldsymbol{x}_{i} = 1 \mid \mathcal{I}) + (1 - \boldsymbol{p}_{i}) \log(1 - p_{\theta}(\boldsymbol{x}_{i} = 1 \mid \mathcal{I}))). \tag{4}$$

### <span id="page-4-1"></span>4.2 CORRECTION STEP

The correction step aims to improve the partial solutions by identifying and discarding the inaccurate predicted variable values that were inappropriately fixed. Specifically, we (1) leverage a trust-region search on the partial solution for a refined solution as a reference for subsequent operations, (2) introduce a novel uncertainty-based metric to determine which subset of variables to fix, and (3) enforce the selected variables to fixed values for dimension reduction.

To begin with, we establish the following notations. Given a MILP instance I, let q(x | I) represent the distribution of the optimal solution, and q(x | xˆ, I) denote the distribution of the reference solution given instance I and predicted solution xˆ. The notation x[P] implies that the partial solution x[P] has the same variable values as x in the index set P.

Trust-Region Search We leverage the solver as a corrector to improve the predicted solutions via trust-region search. Formally, given the predicted marginal probabilities pθ(x | I), we solve the MILP problem [2](#page-3-1) with predefined hyperparameters (k0, k1, △), which is similar to the search process in PS. In this process, the partial solution to be fixed during the search is xˆ[P]. The best primal solution x˜ ∼ q(x | xˆ, I) found by the solver has values x˜[P] for the variable index set P.

Correction Criterion The solution x˜ obtained through the trust-region search serves as a reference to improve the solution quality, called the reference solution. Then, we need to determine which variables to fix and the values they should be assigned. To evaluate the reliability of the predictions for each variable, a natural approach is to compute the distributional discrepancy between the optimal and predicted solutions, specifically DKL (pθ(x<sup>i</sup> | I)||q(x<sup>i</sup> | I)), where we have assumed the independence between different variables. Here the (conditional) Kullback–Leibler (KL) divergence is defined to be DKL(p||q) = P k p(yk) log <sup>p</sup>(yk) q(yk) for distributions p, q and variable y taking values in {y1, y2, · · · , yk, · · · }. However, during testing, the optimal solution is not available, rendering the computation of the KL divergence intractable. Fortunately, we propose an upper bound to estimate the KL divergence that utilizes the available reference solutions.

<span id="page-5-3"></span>Proposition 1 (Uncertainty-Based Error Upper Bound). *We derive the following upper bound for the KL divergence between the predicted marginal probability* pθ(x<sup>i</sup> | I) *and optimal solution distribution* q(x<sup>i</sup> | I)*, utilizing* pθ(x<sup>i</sup> | I) *and the reference solution distribution* q(x<sup>i</sup> | xˆ<sup>i</sup> , I)*,*

<span id="page-5-0"></span>
$$\underbrace{D_{KL}\left(p_{\boldsymbol{\theta}}(\boldsymbol{x}_{i}\mid\mathcal{I})||q(\boldsymbol{x}_{i}\mid\mathcal{I})\right)}_{Target\ Distance}\leq\underbrace{\mathcal{H}\left(p_{\boldsymbol{\theta}}(\boldsymbol{x}_{i}\mid\mathcal{I})\right)}_{Prediction\ Uncertainty}+\underbrace{d\left(p_{\boldsymbol{\theta}}(\boldsymbol{x}_{i}\mid\mathcal{I}),q(\boldsymbol{x}_{i}\mid\hat{\boldsymbol{x}}_{i},\mathcal{I})\right)}_{Prediction\ Correction\ Discrepancy},\tag{5}$$

*where* H(·) *denotes the entropy with* H(p) = − P k p(yk)log(p(yk)) *for variable* y *taking values in* {y1, y2, · · · , yk, · · · }*, and* d(·, ·) *represents the (conditional) cross-entropy loss of distributions with* d(p, q) = − P k q(yk) log(p(yk))*.*

We define the upper bound in Equation [\(5\)](#page-5-0) as the Uncertainty-based Error upper BOund (UEBO), represented as UEBO(p, q) := H(p)+d(p, q) for distributions p and q. The first term H(pθ(x<sup>i</sup> | I)) on the right-hand side of Equation [\(5\)](#page-5-0), referred to as prediction uncertainty, reflects the confidence of the predictor in its predictions. A lower negative entropy value indicates lower uncertainty and greater confidence in the predictor pθ. The second term d(pθ(x<sup>i</sup> | I), q(x<sup>i</sup> | xˆ<sup>i</sup> , I)), called the prediction-correction discrepancy, quantifies the divergence between the predicted and reference solutions. A larger discrepancy suggests that further scrutiny of the predicted results is necessary. We will now discuss why UEBO has the potential to be an effective metric for selecting which variables to fix.

- 1. Providing an upper bound of the intractable KL divergence DKL (pθ(x<sup>i</sup> | I)||q(x<sup>i</sup> | I)). During testing, the distribution of the optimal solution q(x<sup>i</sup> | I) is generally unknown, making the computation of this KL divergence intractable. Instead, UEBO offers a practical estimation by utilizing the available distributions pθ(x<sup>i</sup> | I), q(x<sup>i</sup> | xˆ<sup>i</sup> , I).
- 2. Estimating the discrepancy between the solutions. UEBO aims to penalize variables that exhibit substantial prediction uncertainty or significant disagreement, indicating the reliability of the predicted values.

Problem Reduction. We begin by selecting variables with low UEBO according to the correction rule, as low UEBO indicates higher reliability and greater potential for high-quality solutions. Consequently, we can be more confident in fixing these variables to construct a partial solution x Corr[P ′ ] = F(xˆ[P], x˜[P]), referred to as the corrected partial solution. Specifically, the correction operator F takes in the predicted and reference partial solutions xˆ[P] and x˜[P] and identifies a new index set P ′ ⊂ P of variables to fix, along with their corresponding fixed values. Finally, we arrive at the following reduced problem for the next iteration.

<span id="page-5-1"></span>
$$\min_{\boldsymbol{x} \in \mathbb{R}^n} \quad \boldsymbol{c}^\top \boldsymbol{x}, \quad \text{s.t.} \quad \boldsymbol{A} \boldsymbol{x} \leq \boldsymbol{b}, \boldsymbol{l} \leq \boldsymbol{x} \leq \boldsymbol{u}, \\
\boldsymbol{x}[P] = \boldsymbol{x}^{Corr}[P'], \quad \boldsymbol{x} \in \mathbb{Z}^p \times \mathbb{R}^{n-p}.$$
(6)

Furthermore, to accelerate the convergence, we can introduce a cut c <sup>⊤</sup>x < c <sup>⊤</sup>x˜ into the reduced problem to ensure monotonic improvement.

#### <span id="page-5-2"></span>4.3 ANALYSIS OF THE FIXING STRATEGY

This part is organized as follows. (1) We begin by introducing the concept of prediction-correction consistency for a variable and illustrating its close relationship with UEBO. (2) We propose a straightforward strategy F for approximating UEBO and fixing variables. (3) We analyze the advancement properties of Apollo-MILP incorporated with the proposed fixing strategy.

(1) UEBO and Prediction-Correction Consistency To provide deeper insight into UEBO, we first introduce the concept of prediction-correction consistency as follows.

<span id="page-6-5"></span>Definition 1. We call a variable x<sup>i</sup> prediction-correction consistent if the predicted and reference partial solutions yield the same variable value, i.e., xˆ<sup>i</sup> = x˜<sup>i</sup> . Furthermore, we define the predictioncorrection consistency of a variable as the negative of the prediction-correction discrepancy, given by −d(pθ(x<sup>i</sup> | I), q(x<sup>i</sup> | xˆ<sup>i</sup> , I)).

We investigate the relation between UEBO and prediction-correction consistency. Our findings indicate that prediction-correction consistency serves as a useful estimator of UEBO, as demonstrated in Theorem [2,](#page-6-1) with the proof available in Appendix [C.2.](#page-19-0)

<span id="page-6-1"></span>Theorem 2 (UEBO and Consistency). *Given a variable* x<sup>i</sup> *, UEBO is monotonically increasing with respect to the prediction-correction discrepancy. Therefore, UEBO decreases as the predictioncorrection consistency increases.*

Theorem [2](#page-6-1) illustrates that to compare the UEBOs of two variables, it suffices to compare their prediction-correction consistencies.

(2) Variable Fixing Strategy We define the following consistency-based variable fixing strategy given the predicted and reference partial solutions, xˆ[P] and x˜[P]. Specifically, we let

<span id="page-6-2"></span>
$$P' = \{ i \in P \mid \hat{x}_i = \tilde{x}_i \}, \quad \mathbf{x}^{Corr}[P'] := \hat{\mathbf{x}}[P'] = \tilde{\mathbf{x}}[P']. \tag{7}$$

The fixing strategy outlined in Equation [\(7\)](#page-6-2) fixes the variables that are prediction-correction consistent. We will demonstrate the significant advantages of this proposed variable fixing strategy compared to those based solely on predicted or reference solutions, emphasizing its ability to further enhance precision. We present the pseudo-code of Apollo-MILP in Algorithm [1.](#page-7-0)

(3) Advancement of the Fixing Strategy Let q(x<sup>i</sup> | x˜<sup>i</sup> , xˆ<sup>i</sup> , I) be the marginal distribution of the optimal solution for variable x<sup>i</sup> , given the predicted value xˆ<sup>i</sup> and reference values x˜<sup>i</sup> . We outline the following consistency conditions. The condition is motivated by a classical probabilistic problem: two students provide the same answer to a multiple-choice question respectively, then the answer is more likely to be correct (see Appendix [B.1](#page-17-0) for more details). Analogous to the problem, the condition is intuitive and straightforward: we have greater confidence in the precision of the prediction q(x<sup>i</sup> = 1 | x˜<sup>i</sup> = 1, xˆ<sup>i</sup> = 1, I) for the optimal variable value x ∗ <sup>i</sup> when the predicted and reference values, xˆ<sup>i</sup> and x˜<sup>i</sup> , yield the same result.

<span id="page-6-4"></span>Assumption 1 (Consistency Conditions). *Consistency between the predicted and reference values for variable* x<sup>i</sup> *enhances the likelihood of precisely predicting the optimal solution, i.e.,*

<span id="page-6-3"></span>
$$q(\boldsymbol{x}_{i}=1 \mid \tilde{\boldsymbol{x}}_{i}=1, \hat{\boldsymbol{x}}_{i}=1, \mathcal{I}) \geq q(\boldsymbol{x}_{i}=1 \mid \tilde{\boldsymbol{x}}_{i}=0, \hat{\boldsymbol{x}}_{i}=1, \mathcal{I}), \text{ and}$$

$$q(\boldsymbol{x}_{i}=1 \mid \tilde{\boldsymbol{x}}_{i}=1, \hat{\boldsymbol{x}}_{i}=1, \mathcal{I}) \geq q(\boldsymbol{x}_{i}=1 \mid \tilde{\boldsymbol{x}}_{i}=1, \hat{\boldsymbol{x}}_{i}=0, \mathcal{I}).$$
(8)

Based on the above conditions, we analyze the effects of fixing the prediction-correction consistent variables. The proposed strategy ensures greater precision in identifying the optimal variable value.

<span id="page-6-0"></span>Theorem 3 (Precision Improvement Guarantee). *Suppose the consistency conditions [\(8\)](#page-6-3) hold. Then, the prediction precision for variables with consistent results* q(x<sup>i</sup> = 1 | x˜<sup>i</sup> = 1, xˆ<sup>i</sup> = 1, I) *is higher than that of variables based solely on the predicted or reference solutions, i.e.,*

$$q(\mathbf{x}_{i} = 1 \mid \tilde{\mathbf{x}}_{i} = 1, \hat{\mathbf{x}}_{i} = 1, \mathcal{I}) \ge q(\mathbf{x}_{i} = 1 \mid \tilde{\mathbf{x}}_{i} = 1, \mathcal{I}), \text{ and}$$

$$q(\mathbf{x}_{i} = 1 \mid \tilde{\mathbf{x}}_{i} = 1, \hat{\mathbf{x}}_{i} = 1, \mathcal{I}) \ge q(\mathbf{x}_{i} = 1 \mid \hat{\mathbf{x}}_{i} = 1, \mathcal{I}).$$
(9)

Please refer to Appendix [C.3](#page-19-1) for the proof. Finally, we examine the feasibility guarantee of Apollo-MILP. As the feasibility of PS is closely related to Problem [2,](#page-3-1) we show that our method allows for better feasibility than Problem [2,](#page-3-1) and hence the PS method.

Corollary 4. *Suppose we select variables to fix based on the strategy in Equation [\(7\)](#page-6-2). If the trustregion searching problem [2](#page-3-1) (the PS method) is feasible, then the corresponding reduced problem [6](#page-5-1) provided by Apollo-MILP will also be feasible.*

### 5 EXPERIMENTS

In this part, we conduct extensive studies to demonstrate the effectiveness of our framework. Our method achieves significant improvements in solving performance (Section [5.2\)](#page-8-0), generalization ability (Appendix [H.5\)](#page-26-0), and real-world applicability (Appendix [H.1\)](#page-23-0). Please refer to Appendix [D](#page-20-1) for a detailed implementation of the methods.

Algorithm 1: Alternating Prediction-Correction Neural Solving Framework

```
Input: MILP Instance I to solve, the predictor pθ, iteration number K, hyperparameters
          {(k
             (i)
             0
                , k(i)
                  1
                     , △(i)
                          )}
                            K
                            i=1
1 Initialize: the reduced problem I
                                         (0) ← I.
2 for k in {0, · · · , K} do
3 # Prediction Step
4 Obtain a predicted solution xˆ ∼ pθ(x | I(k)
                                                  ) from the predictor pθ
5 Determine the partial solution xˆ[P] ∈ xˆ to fix according to (k
                                                                   (k)
                                                                   0
                                                                      , k(k)
                                                                        1
                                                                           )
6 # Correction Step
7 Construct the trust-region searching Problem 2 over I
                                                           (k) using (k
                                                                       (k)
                                                                       0
                                                                          , k(k)
                                                                            1
                                                                               , △(k)
                                                                                     )
8 if k=K then
 9 Leveraging a solver to solve the Problem 2 for the best solution x˜
                                                                          ∗
10 end
11 else
12 Leveraging a solver to solve the Problem 2 for a reference solution x˜ ∼ q(x | xˆ, I
                                                                                           (k)
                                                                                              )
13 Obtain x
                   Corr[P
                          ′
                           ] using Criterion (7)
14 Fix x
                Corr[P
                       ′
                       ] in I
                             (k)
                                to obtain the new reduced problem I
                                                                    (k+1)
15 end
16 end
17 return the best solution x˜
                            ∗
```

### <span id="page-7-1"></span>5.1 EXPERIMENT SETTINGS

Benchmarks We conduct experiments on four popular MILP benchmarks utilized in the ML4CO field: combinatorial auctions (CA) [\(Leyton-Brown et al., 2000\)](#page-13-11), set covering (SC) [\(Balas & Ho,](#page-10-4) [1980\)](#page-10-4), item placement (IP) [\(Gasse et al., 2022\)](#page-11-0) and workload appointment (WA) [\(Gasse et al.,](#page-11-0) [2022\)](#page-11-0). The first two benchmarks are standard benchmarks proposed in [\(Gasse et al., 2019\)](#page-11-1) and are commonly used to evaluate the performance of ML solvers [\(Gasse et al., 2019;](#page-11-1) [Han et al., 2023;](#page-12-0) [Huang et al., 2024\)](#page-12-1). The last two benchmarks, IP and WA, come from two challenging real-world problem families used in NeurIPS ML4CO 2021 competition [\(Gasse et al., 2022\)](#page-11-0). We use 240 training, 60 validation, and 100 testing instances, following the settings in [Han et al.](#page-12-0) [\(2023\)](#page-12-0). Please refer to Appendix [F](#page-21-0) for more details on the benchmarks.

Baselines We consider the following baselines in our experiments. We compare the proposed method with Neural Diving (ND) [\(Nair et al., 2020\)](#page-14-0) and Predict-and-Search (PS) [\(Han et al., 2023\)](#page-12-0), which we have introduced in the previous sections. Contrastive Predict-and-Search (ConPS) [\(Huang](#page-12-1) [et al., 2024\)](#page-12-1) is a strong baseline, leveraging contrastive learning to enhance the performance of PS. For ConPS, we set the ratio of positive to negative samples at ten, using low-quality solutions as negative samples. These baselines operate independently of the backbone solvers and can be integrated with traditional solvers such as SCIP [\(Achterberg, 2009\)](#page-10-2) and Gurobi [\(Gurobi Optimization,](#page-12-2) [2021\)](#page-12-2). Therefore, we also include SCIP and Gurobi as baselines for a comprehensive comparison. Following [Han et al.](#page-12-0) [\(2023\)](#page-12-0), Gurobi and SCIP are set to focus on finding better primal solutions.

Metrics We evaluate the methods on each test instance and record the best objective value OBJ within 1,000 seconds. Following the setting in [Han et al.](#page-12-0) [\(2023\)](#page-12-0), we also run a single-thread Gurobi for 3,600 seconds and denote the best objective value as the best-known solution (BKS) to approximate the optimal value. However, we find that our method, when built on Gurobi, can identify better solutions within 1,000 seconds than Gurobi achieves in 3,600 seconds for the IP and WA benchmarks. As a result, we use the best objectives obtained by our approach as the BKS for these two benchmarks. We define the absolute primal gap as the difference between the best objective found by the solvers and the BKS, expressed as gapabs := |OBJ − BKS|. Within the same solving time, a lower absolute primal gap indicates stronger performance.

Implementations In our experiments, we conduct four rounds of iterations. The time allocated for each iteration is 100, 100, 200, and 600 seconds, respectively. We denote the size of the partial solution in the i th iteration by k (i) = k (i) <sup>0</sup> +k (i) <sup>1</sup> with k (i) 0 variables fixed to 0 and k (i) 1 to 1, and allow

<span id="page-8-1"></span>Table 1: Comparison of solving performance between our approach and baseline methods, under a 1, 000s time limit. We build the ML approaches on Gurobi and SCIP, respectively. As we choose the challenging benchmarks with large-size instances, the solvers reach the time limit in all the experiments. We thus report the average best objective values and the absolute primal gap. '↑' indicates that higher is better, and '↓' indicates that lower is better. We mark the best values in bold. We also report the improvement of our method over the traditional solvers in terms of gapabs. We find our method with a 1,000s runtime can outperform Gurobi with 3,600s runtime in IP and WA.

|                               | CA (BKS 97616.59)             |                      |        | SC (BKS 122.95) |                 | IP (BKS 8.90)      |        | WA (BKS 704.88)         |  |
|-------------------------------|-------------------------------|----------------------|--------|-----------------|-----------------|--------------------|--------|-------------------------|--|
|                               | Obj ↑                         | ↓<br>gapabs          | Obj ↓  | ↓<br>gapabs     | Obj ↓           | ↓<br>gapabs        | Obj ↓  | ↓<br>gapabs             |  |
| Gurobi                        | 97297.52                      | 319.07               | 123.40 | 0.45            | 9.38            | 0.48               | 705.49 | 0.61                    |  |
| ND+Gurobi                     | 96002.99                      | 1613.59              | 123.25 | 0.29            | 9.33            | 0.43               | 705.70 | 0.82                    |  |
| PS+Gurobi                     | 97358.23                      | 258.36               | 123.30 | 0.35            | 9.17            | 0.27               | 705.45 | 0.57                    |  |
| ConPS+Gurobi                  | 97464.10                      | 152.49               | 123.20 | 0.25            | 9.09            | 0.19               | 705.37 | 0.49                    |  |
| Ours+Gurobi                   | 97487.18                      | 129.41               | 123.05 | 0.10            | 8.90            | 0.00               | 704.88 | 0.00                    |  |
| Improvement                   |                               | 52.2%                |        | 77.8%           |                 | 100.0%             |        | 100.0%                  |  |
| SCIP                          | 96544.10                      | 1072.48              | 124.80 | 1.85            | 14.50           | 5.60               | 709.62 | 4.74                    |  |
| ND+SCIP                       | 95909.50                      | 1707.09              | 123.90 | 0.95            | 13.61           | 4.71               | 709.55 | 4.67                    |  |
| PS+SCIP                       | 96783.62                      | 832.97               | 124.35 | 1.40            | 14.25           | 5.35               | 709.39 | 4.51                    |  |
| ConPS+SCIP                    | 96824.26                      | 792.33               | 123.90 | 0.95            | 13.74           | 4.84               | 709.33 | 4.45                    |  |
| Ours+SCIP                     | 96839.34                      | 777.25               | 123.50 | 0.55            | 12.86           | 3.96               | 709.29 | 4.41                    |  |
| Improvement                   |                               | 27.5%                |        | 70.2%           |                 | 29.2%              |        | 6.9%                    |  |
| CA                            |                               | SC                   |        |                 | IP              |                    |        | WA                      |  |
| Average Primal Gap<br>2<br>10 | 1<br>10<br>2<br>10<br>3<br>10 |                      |        | 1<br>10         |                 | 1<br>10<br>3<br>10 |        |                         |  |
| 0<br>500<br>Time (s)          | 1000                          | 0<br>500<br>Time (s) | 1000   | 0               | 500<br>Time (s) | 1000               | 0      | 500<br>1000<br>Time (s) |  |

<span id="page-8-2"></span>Figure 3: The primal gap of the approaches as the solving process proceeds. Our methods are implemented using Gurobi, with a time limit set to 1,000s, and we average the results across 100 testing instances. A lower primal gap for our method indicates stronger convergence performance.

Gurobi ND+Gurobi PS+Gurobi ConPS+Gurobi Ours+Gurobi

△(i) of the fixed variables to be flipped during the trust-region search. The total size of the partial solutions is given by kfix = P<sup>4</sup> <sup>i</sup>=1 k (i) , which sums the partial solution sizes across all iterations. More details on hyperparameters are in Appendix [G.](#page-22-0)

### <span id="page-8-0"></span>5.2 MAIN EVALUATION

Solving Performance To evaluate the effectiveness of the proposed method, we compare the solving performance between our framework and the baselines, under a time limit of 1,000 seconds. Table [1](#page-8-1) presents the average best objectives found by the solvers alongside the average absolute primal gap. The instances in the IP and WA datasets possess more complex structures and larger sizes, making them more challenging for the solvers. While ND demonstrates strong performance in the CA and SC datasets, it falls short in the real-world datasets, IP and WA. ConPS serves as a robust baseline across all benchmarks, indicating that contrastive learning effectively enhances the predictor, leading to higher-quality predicted solutions. The results reveal that our proposed Apollo-MILP consistently outperforms the baselines, achieving the best objectives and the lowest gaps across the benchmarks. Specifically, Apollo-MILP reduces the absolute primal gap by over 80% compared to Gurobi and by 30% compared to SCIP. Furthermore, in the IP and WA benchmarks, our approach identifies better solutions within 1,000s than those obtained by running Gurobi for 3,600s.

Primal Gap as a Function of Runtime Figure [3](#page-8-2) illustrates the curves of the average primal gap, defined as gaprel := |OBJ − BKS|/|BKS|, throughout the solving process. Similar to the absolute

primal gap, the primal gap reflects the convergence properties of the solvers; a rapid decrease in the curves indicates superior solving performance. As shown in Figure [3,](#page-8-2) the primal gap of Apollo-MILP exhibits a gradual decrease in the early stages as it focuses on correction steps to improve the quality of partial solutions. Subsequently, the primal gap demonstrates a rapid decline, ultimately achieving the lowest gap, which highlights Appolo-MILP's strong convergence performance.

#### 5.3 ABLATION STUDY

Fixing Strategies To better understand Apollo-MILP, we conduct ablation studies on the variable fixing strategies. Specifically, we implement two baselines for variable fixing strategies: Direct Fixing (four rounds of direct fixing) and Multi-stage PS (four rounds of PS). We utilize the same set of hyperparameters as our method. The Multi-stage PS strategy directly fixes the variables in the predicted partial solutions xˆ[P]. The Direct Fixing strategy directly fixes the variables in the reference partial solu-

<span id="page-9-0"></span>Table 2: Comparison of solving performance between our approach and different fixing strategies, under a 1, 000s time limit. We report the average best objective values and absolute primal gap. '↓' indicates that lower is better. We mark the best values in bold.

|                | IP (BKS 8.90) |             | WA (BKS 704.88) |             |
|----------------|---------------|-------------|-----------------|-------------|
|                | Obj ↓         | ↓<br>gapabs | Obj ↓           | ↓<br>gapabs |
| Gurobi         | 9.38          | 0.48        | 705.49          | 0.61        |
| PS+Gurobi      | 9.17          | 0.27        | 705.45          | 0.57        |
| Direct Fixing  | 9.22          | 0.32        | 705.40          | 0.52        |
| Multi-stage PS | 9.18          | 0.28        | 705.33          | 0.45        |
| Ours+Gurobi    | 8.90          | 0.00        | 704.88          | 0.00        |

tions x˜[P]. The results on the IP and WA benchmarks are presented in Table [2.](#page-9-0) The results in Table [2](#page-9-0) show that our proposed consistency-based fixing strategy outperforms the other baselines, highlighting our method's effectiveness. Please see Appendix [H.3](#page-25-0) for more experiment results.

#### Comparison with Warm-Starting Gurobi Warm-starting is an alternative to the trust-region search in PS and our method, in which we provide an initial feasible solution to Gurobi to guide the solving process. Gurobi can search around these start solutions or partial solutions. Warm-starting is a crucial baseline to help us understand the trust-region search. Specifically, we implement two methods, warm-starting PS (WS-PS) and warmstarting our method (WS-Ours). WS-PS passes the initial GNN prediction to Gurobi as a start solution, with hyperparameters such as k<sup>0</sup> and k<sup>1</sup> same as those we conduct in our main ex-

Table 3: Comparison of solving performance between our method and the warm-starting methods, under a 1, 000s time limit. We report the average best objective values and absolute primal gap. '↓' indicates that lower is better. We mark the best values in bold.

|                               | IP (BKS 8.90) |              | WA (BKS 704.88)  |              |
|-------------------------------|---------------|--------------|------------------|--------------|
|                               | Obj ↓         | ↓<br>gapabs  | Obj ↓            | ↓<br>gapabs  |
| Gurobi                        | 9.38          | 0.48         | 705.49           | 0.61         |
| PS+Gurobi                     | 9.17          | 0.27         | 705.45           | 0.57         |
| WS-PS+Gurobi                  | 9.20          | 0.30         | 705.45           | 0.57         |
| WS-Ours+Gurobi<br>Ours+Gurobi | 9.13<br>8.90  | 0.23<br>0.00 | 705.40<br>704.88 | 0.52<br>0.00 |

periments. We also implement WS-Ours, which employs the same prediction model but replaces the trust-region search with warm-starting at each step. The results are presented in Table [19,](#page-26-1) in which we set the solving time limit as 1,000s. The results show that WS Gurobi performs comparably to PS, while WS Ours combined with Gurobi outperforms WS Gurobi, demonstrating the effectiveness of our proposed variable fixing strategy. Finally, our proposed method performs the best. The trust-region search is a more effective search method that aligns well with our framework. Please see Appendix [H.3](#page-25-0) for more experiment results.

### 6 CONCLUSION AND FUTURE WORKS

In this paper, we propose a novel ML-based solving framework (Apollo-MILP) to identify highquality solutions for MILP problems. Apollo-MILP leverages the strengths of both Neural Diving and Predict-and-Search, alternating between prediction and correction steps to iteratively refine the predicted solutions and reduce the complexity of MILP problems. Experiments show that Apollo-MILP significantly outperforms other ML-based approaches in terms of solution quality, demonstrating strong generalization ability and promising real-world applicability.

#### ACKNOWLEDGMENTS

The authors would like to thank all the anonymous reviewers for their valuable suggestions. This work was supported by the National Key R&D Program of China under contract 2022ZD0119801 and the National Nature Science Foundations of China grants U23A20388 and 62021001.

# ETHIC STATEMENT

This paper aims to explore the potential of an efficient MILP solving framework and obey the ICLR code of ethics. We do not foresee any direct, immediate, or negative societal impacts stemming from the outcomes of our research.

### REPRODUCIBILITY STATEMENT

We provide the following information for the reproducibility of our proposed Apollo-MILP.

- 1. Method. We provide the pseudo-code of our method in Section [4.3.](#page-5-2) Moreover, we will make our source code publicly available once the paper is accepted for publication.
- 2. Theoretical Proof. We provide the proof of our theoretical results in Appendix [C.](#page-19-2)
- 3. Implementations. We discuss the hyperparameters in Table [10](#page-23-1) of Appendix [G.](#page-22-0) The information on the implementation details can be found in Appendix [D.](#page-20-1)

# 7 ACKNOWLEDGEMENT

The authors would like to thank all the anonymous reviewers for their insightful comments and valuable suggestions. This work was supported by the National Key R&D Program of China under contract 2022ZD0119801 and the National Nature Science Foundations of China grants U23A20388 and 62021001.

### REFERENCES

<span id="page-10-2"></span>Tobias Achterberg. Scip: solving constraint integer programs. *Mathematical Programming Computation*, 1:1–41, 2009.

<span id="page-10-5"></span>Yinqi Bai, Jie Wang, Lei Chen, Zhihai Wang, Yufei Kuang, Mingxuan Yuan, Jianye HAO, and Feng Wu. A graph enhanced symbolic discovery framework for efficient circuit synthesis. In *The Thirteenth International Conference on Learning Representations*, 2025. URL [https:](https://openreview.net/forum?id=EG9nDN3eGB) [//openreview.net/forum?id=EG9nDN3eGB](https://openreview.net/forum?id=EG9nDN3eGB).

<span id="page-10-4"></span>Egon Balas and Andrew Ho. *Set covering algorithms using cutting planes, heuristics, and subgradient optimization: a computational study*. Springer, 1980.

<span id="page-10-3"></span>Maria-Florina F Balcan, Siddharth Prasad, Tuomas Sandholm, and Ellen Vitercik. Structural analysis of branch-and-cut and the learnability of gomory mixed integer cuts. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), *Advances in Neural Information Processing Systems*, volume 35, pp. 33890–33903. Curran Associates, Inc., 2022. URL [https://proceedings.neurips.cc/paper\\_files/paper/2022/](https://proceedings.neurips.cc/paper_files/paper/2022/file/db2cbf43a349bc866111e791b58c7bf4-Paper-Conference.pdf) [file/db2cbf43a349bc866111e791b58c7bf4-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/db2cbf43a349bc866111e791b58c7bf4-Paper-Conference.pdf).

<span id="page-10-1"></span>Yoshua Bengio, Andrea Lodi, and Antoine Prouvost. Machine learning for combinatorial optimization: a methodological tour d'horizon. *European Journal of Operational Research*, 290(2): 405–421, 2021.

<span id="page-10-0"></span>Robert E Bixby, Mary Fenelon, Zonghao Gu, Ed Rothberg, and Roland Wunderling. Mixed-integer programming: A progress report. In *The sharpest cut: the impact of Manfred Padberg and his work*, pp. 309–325. SIAM, 2004.

- <span id="page-11-8"></span>Junyang Cai, Taoan Huang, and Bistra Dilkina. Learning backdoors for mixed integer programs with contrastive learning. In *The 27th European Conference on Artificial Intelligence*, 2024.
- <span id="page-11-5"></span>Ziang Chen, Jialin Liu, Xinshang Wang, and Wotao Yin. On representing linear programs by graph neural networks. In *The Eleventh International Conference on Learning Representations*, 2023a.
- <span id="page-11-6"></span>Ziang Chen, Jialin Liu, Xinshang Wang, and Wotao Yin. On representing mixed-integer linear programs by graph neural networks. In *The Eleventh International Conference on Learning Representations*, 2023b.
- <span id="page-11-3"></span>Antonia Chmiela, Elias Khalil, Ambros Gleixner, Andrea Lodi, and Sebastian Pokutta. Learning to schedule heuristics in branch and bound. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), *Advances in Neural Information Processing Systems*, volume 34, pp. 24235–24246. Curran Associates, Inc., 2021. URL [https://proceedings.neurips.cc/paper\\_files/paper/2021/](https://proceedings.neurips.cc/paper_files/paper/2021/file/cb7c403aa312160380010ee3dd4bfc53-Paper.pdf) [file/cb7c403aa312160380010ee3dd4bfc53-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2021/file/cb7c403aa312160380010ee3dd4bfc53-Paper.pdf).
- <span id="page-11-7"></span>Jian-Ya Ding, Chao Zhang, Lei Shen, Shengyin Li, Bing Wang, Yinghui Xu, and Le Song. Accelerating primal solution findings for mixed integer programs based on solution prediction. In *The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020*, pp. 1452–1459. AAAI Press, 2020. doi: 10.1609/AAAI.V34I02.5503. URL <https://doi.org/10.1609/aaai.v34i02.5503>.
- <span id="page-11-10"></span>Huanshuo Dong, Hong Wang, Haoyang Liu, Jian Luo, Jie Wang, et al. Accelerating pde data generation via differential operator action in solution space. In *Forty-first International Conference on Machine Learning*, 2025.
- <span id="page-11-1"></span>Maxime Gasse, Didier Chetelat, Nicola Ferroni, Laurent Charlin, and Andrea Lodi. Exact combi- ´ natorial optimization with graph convolutional neural networks. *Advances in neural information processing systems*, 32, 2019.
- <span id="page-11-0"></span>Maxime Gasse, Simon Bowly, Quentin Cappart, Jonas Charfreitag, Laurent Charlin, Didier Chetelat, ´ Antonia Chmiela, Justin Dumouchelle, Ambros Gleixner, Aleksandr M. Kazachkov, Elias Khalil, Pawel Lichocki, Andrea Lodi, Miles Lubin, Chris J. Maddison, Morris Christopher, Dimitri J. Papageorgiou, Augustin Parjadis, Sebastian Pokutta, Antoine Prouvost, Lara Scavuzzo, Giulia Zarpellon, Linxin Yang, Sha Lai, Akang Wang, Xiaodong Luo, Xiang Zhou, Haohan Huang, Shengcheng Shao, Yuanming Zhu, Dong Zhang, Tao Quan, Zixuan Cao, Yang Xu, Zhewei Huang, Shuchang Zhou, Chen Binbin, He Minggui, Hao Hao, Zhang Zhiyu, An Zhiwu, and Mao Kun. The machine learning for combinatorial optimization competition (ml4co): Results and insights. In Douwe Kiela, Marco Ciccone, and Barbara Caputo (eds.), *Proceedings of the NeurIPS 2021 Competitions and Demonstrations Track*, volume 176 of *Proceedings of Machine Learning Research*, pp. 220–231. PMLR, 06–14 Dec 2022. URL [https:](https://proceedings.mlr.press/v176/gasse22a.html) [//proceedings.mlr.press/v176/gasse22a.html](https://proceedings.mlr.press/v176/gasse22a.html).
- <span id="page-11-4"></span>Zijie Geng, Xijun Li, Jie Wang, Xiao Li, Yongdong Zhang, and Feng Wu. A deep instance generative framework for milp solvers under limited data availability. In *Advances in Neural Information Processing Systems*, 2023.
- <span id="page-11-9"></span>Zijie Geng, Jie Wang, Xijun Li, Fangzhou Zhu, Jianye HAO, Bin Li, and Feng Wu. Differentiable integer linear programming. In *The Thirteenth International Conference on Learning Representations*, 2025. URL <https://openreview.net/forum?id=FPfCUJTsCn>.
- <span id="page-11-11"></span>Ambros Gleixner, Gregor Hendel, Gerald Gamrath, Tobias Achterberg, Michael Bastubbe, Timo Berthold, Philipp Christophel, Kati Jarck, Thorsten Koch, Jeff Linderoth, et al. Miplib 2017: datadriven compilation of the 6th mixed-integer programming library. *Mathematical Programming Computation*, 13(3):443–490, 2021.
- <span id="page-11-2"></span>Prateek Gupta, Maxime Gasse, Elias Khalil, Pawan Mudigonda, Andrea Lodi, and Yoshua Bengio. Hybrid models for learning to branch. *Advances in neural information processing systems*, 33: 18087–18097, 2020.

- <span id="page-12-4"></span>Prateek Gupta, Elias B Khalil, Didier Chetelat, Maxime Gasse, Yoshua Bengio, Andrea Lodi, and ´ M Pawan Kumar. Lookback for learning to branch. *arXiv preprint arXiv:2206.14987*, 2022.
- <span id="page-12-2"></span>LLC Gurobi Optimization. Gurobi optimizer. *URL http://www. gurobi. com*, 2021.
- <span id="page-12-0"></span>Qingyu Han, Linxin Yang, Qian Chen, Xiang Zhou, Dong Zhang, Akang Wang, Ruoyu Sun, and Xiaodong Luo. A gnn-guided predict-and-search framework for mixed-integer linear programming. In *The Eleventh International Conference on Learning Representations*, 2023.
- <span id="page-12-7"></span>He He, Hal Daume III, and Jason M Eisner. Learning to search in branch and bound algorithms. *Advances in neural information processing systems*, 27, 2014.
- <span id="page-12-10"></span>Taoan Huang, Aaron Ferber, Yuandong Tian, Bistra N. Dilkina, and Benoit Steiner. Searching large neighborhoods for integer linear programs with contrastive learning. In *International Conference on Machine Learning*, 2023. URL [https://api.semanticscholar.org/CorpusID:](https://api.semanticscholar.org/CorpusID:256598329) [256598329](https://api.semanticscholar.org/CorpusID:256598329).
- <span id="page-12-1"></span>Taoan Huang, Aaron M Ferber, Arman Zharmagambetov, Yuandong Tian, and Bistra Dilkina. Contrastive predict-and-search for mixed integer linear programs. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pp. 19757–19771. PMLR, 21–27 Jul 2024. URL <https://proceedings.mlr.press/v235/huang24f.html>.
- <span id="page-12-6"></span>Zeren Huang, Kerong Wang, Furui Liu, Hui-Ling Zhen, Weinan Zhang, Mingxuan Yuan, Jianye Hao, Yong Yu, and Jun Wang. Learning to select cuts for efficient mixed-integer programming. *Pattern Recognition*, 123:108353, 2022. ISSN 0031-3203. doi: https://doi.org/10.1016/j.patcog. 2021.108353. URL [https://www.sciencedirect.com/science/article/pii/](https://www.sciencedirect.com/science/article/pii/S0031320321005331) [S0031320321005331](https://www.sciencedirect.com/science/article/pii/S0031320321005331).
- <span id="page-12-3"></span>Elias B. Khalil, Pierre Le Bodic, Le Song, George Nemhauser, and Bistra Dilkina. Learning to branch in mixed integer programming. In Dale Schuurmans and Michael Wellman (eds.), *Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence*, pp. 724–731, United States of America, 2016. Association for the Advancement of Artificial Intelligence (AAAI). URL <http://www.aaai.org/Conferences/AAAI/aaai16.php>. AAAI Conference on Artificial Intelligence 2016, AAAI 2016 ; Conference date: 12-02-2016 Through 17-02-2016.
- <span id="page-12-9"></span>Elias B. Khalil, Bistra Dilkina, George L. Nemhauser, Shabbir Ahmed, and Yufen Shao. Learning to run heuristics in tree search. In *Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, IJCAI-17*, pp. 659–666, 2017. doi: 10.24963/ijcai.2017/92. URL [https:](https://doi.org/10.24963/ijcai.2017/92) [//doi.org/10.24963/ijcai.2017/92](https://doi.org/10.24963/ijcai.2017/92).
- <span id="page-12-11"></span>Elias B. Khalil, Christopher Morris, and Andrea Lodi. MIP-GNN: A data-driven framework for guiding combinatorial solvers. In *Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022*, pp. 10219–10227. AAAI Press, 2022. doi: 10.1609/AAAI. V36I9.21262. URL <https://doi.org/10.1609/aaai.v36i9.21262>.
- <span id="page-12-12"></span>Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In *International Conference on Learning Representations*, 2017. URL [https:](https://openreview.net/forum?id=SJU4ayYgl) [//openreview.net/forum?id=SJU4ayYgl](https://openreview.net/forum?id=SJU4ayYgl).
- <span id="page-12-5"></span>Yufei Kuang, Jie Wang, Haoyang Liu, Fangzhou Zhu, Xijun Li, Jia Zeng, HAO Jianye, Bin Li, and Feng Wu. Rethinking branching on exact combinatorial optimization solver: The first deep symbolic discovery framework. In *The Twelfth International Conference on Learning Representations*, 2024.
- <span id="page-12-8"></span>Abdel Ghani Labassi, Didier Chetelat, and Andrea Lodi. Learning to compare nodes in branch and bound with graph neural networks. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), *Advances in Neural Information Processing Systems*, volume 35, pp. 32000–32010. Curran Associates, Inc., 2022. URL [https://proceedings.neurips.cc/paper\\_files/paper/2022/](https://proceedings.neurips.cc/paper_files/paper/2022/file/cf5bb18807a3e9cfaaa51e667e18f807-Paper-Conference.pdf) [file/cf5bb18807a3e9cfaaa51e667e18f807-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/cf5bb18807a3e9cfaaa51e667e18f807-Paper-Conference.pdf).

- <span id="page-13-11"></span>Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. Towards a universal test suite for combinatorial auction algorithms. In *Proceedings of the 2nd ACM Conference on Electronic Commerce*, pp. 66–76, 2000.
- <span id="page-13-6"></span>Sirui Li, Wenbin Ouyang, Max B Paulus, and Cathy Wu. Learning to configure separators in branchand-cut. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.
- <span id="page-13-2"></span>Xijun Li, Fangzhou Zhu, Hui-Ling Zhen, Weilin Luo, Meng Lu, Yimin Huang, Zhenan Fan, Zirui Zhou, Yufei Kuang, Zhihai Wang, Zijie Geng, Yang Li, Haoyang Liu, Zhiwu An, Muming Yang, Jianshu Li, Jie Wang, Junchi Yan, Defeng Sun, Tao Zhong, Yong Zhang, Jia Zeng, Mingxuan Yuan, Jianye Hao, Jun Yao, and Kun Mao. Machine learning insides optverse ai solver: Design principles and applications, 2024a.
- <span id="page-13-1"></span>Yixuan Li, Wanyuan Wang, Weiyi Xu, Yanchen Deng, and Weiwei Wu. Factor graph neural network meets max-sum: A real-time route planning algorithm for massive-scale trips. In *Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems*, pp. 1165– 1173, 2024b.
- <span id="page-13-10"></span>Yixuan Li, Can Chen, Jiajun Li, Jiahui Duan, Xiongwei Han, Tao Zhong, Vincent Chau, Weiwei Wu, and Wanyuan Wang. Fast and interpretable mixed-integer linear program solving by learning model reduction. *Proceedings of the AAAI Conference on Artificial Intelligence*, 2025.
- <span id="page-13-3"></span>Jiacheng Lin, Meng XU, Zhihua Xiong, and Huangang Wang. CAMBranch: Contrastive learning with augmented MILPs for branching. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=K6kt50zAiG>.
- <span id="page-13-4"></span>Haotian Ling, Zhihai Wang, and Jie Wang. Learning to stop cut generation for efficient mixedinteger linear programming. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38 (18):20759–20767, Mar. 2024. doi: 10.1609/aaai.v38i18.30064. URL [https://ojs.aaai.](https://ojs.aaai.org/index.php/AAAI/article/view/30064) [org/index.php/AAAI/article/view/30064](https://ojs.aaai.org/index.php/AAAI/article/view/30064).
- <span id="page-13-7"></span>Chang Liu, Zhichen Dong, Haobo Ma, Weilin Luo, Xijun Li, Bowen Pang, Jia Zeng, and Junchi Yan. L2p-MIP: Learning to presolve for mixed integer programming. In *The Twelfth International Conference on Learning Representations*, 2024a. URL [https://openreview.net/](https://openreview.net/forum?id=McfYbKnpT8) [forum?id=McfYbKnpT8](https://openreview.net/forum?id=McfYbKnpT8).
- <span id="page-13-8"></span>Haoyang Liu, Yufei Kuang, Jie Wang, Xijun Li, Yongdong Zhang, and Feng Wu. Promoting generalization for exact solvers via adversarial instance augmentation, 2023a. URL [https:](https://arxiv.org/abs/2310.14161) [//arxiv.org/abs/2310.14161](https://arxiv.org/abs/2310.14161).
- <span id="page-13-9"></span>Haoyang Liu, Jie Wang, Wanbo Zhang, Zijie Geng, Yufei Kuang, Xijun Li, Bin Li, Yongdong Zhang, and Feng Wu. MILP-studio: MILP instance generation via block structure decomposition. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024b. URL <https://openreview.net/forum?id=W433RI0VU4>.
- <span id="page-13-5"></span>Hongyu Liu, Haoyang Liu, Yufei Kuang, Jie Wang, and Bin Li. Deep symbolic optimization for combinatorial optimization: Accelerating node selection by discovering potential heuristics. In *Proceedings of the Genetic and Evolutionary Computation Conference Companion*, pp. 2067– 2075, 2024c.
- <span id="page-13-13"></span>Qiyuan Liu, Qi Zhou, Rui Yang, and Jie Wang. Robust representation learning by clustering with bisimulation metrics for visual reinforcement learning with distractions. In *Thirty-Seventh AAAI Conference on Artificial Intelligence*, pp. 8843–8851. AAAI Press, 2023b.
- <span id="page-13-0"></span>Kefan Ma, Liquan Xiao, Jianmin Zhang, and Tiejun Li. Accelerating an fpga-based sat solver by software and hardware co-design. *Chinese Journal of Electronics*, 28(5):953–961, 2019.
- <span id="page-13-14"></span>Wenyu Mao, Jiancan Wu, Haoyang Liu, Yongduo Sui, and Xiang Wang. Invariant graph learning meets information bottleneck for out-of-distribution generalization, 2024. URL [https:](https://arxiv.org/abs/2408.01697) [//arxiv.org/abs/2408.01697](https://arxiv.org/abs/2408.01697).
- <span id="page-13-12"></span>Wenyu Mao, Shuchang Liu, Haoyang Liu, Haozhe Liu, Xiang Li, and Lantao Hu. Distinguished quantized guidance for diffusion-based sequence recommendation. In *THE WEB CONFERENCE 2025*, 2025a. URL <https://openreview.net/forum?id=L8MeU0K5Fx>.

- <span id="page-14-7"></span>Wenyu Mao, Jiancan Wu, Weijian Chen, Chongming Gao, Xiang Wang, and Xiangnan He. Reinforced prompt personalization for recommendation with large language models, 2025b. URL <https://arxiv.org/abs/2407.17115>.
- <span id="page-14-0"></span>Vinod Nair, Sergey Bartunov, Felix Gimeno, Ingrid Von Glehn, Pawel Lichocki, Ivan Lobov, Brendan O'Donoghue, Nicolas Sonnerat, Christian Tjandraatmadja, Pengming Wang, et al. Solving mixed integer programs using neural networks. *arXiv preprint arXiv:2012.13349*, 2020.
- <span id="page-14-1"></span>Max Paulus and Andreas Krause. Learning to dive in branch and bound. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), *Advances in Neural Information Processing Systems*, volume 36, pp. 34260–34277. Curran Associates, Inc., 2023. URL [https://proceedings.neurips.cc/paper\\_files/paper/2023/](https://proceedings.neurips.cc/paper_files/paper/2023/file/6bbda0824bcc20749f21510fd8b28de5-Paper-Conference.pdf) [file/6bbda0824bcc20749f21510fd8b28de5-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/6bbda0824bcc20749f21510fd8b28de5-Paper-Conference.pdf).
- <span id="page-14-4"></span>Max B. Paulus, Giulia Zarpellon, Andreas Krause, Laurent Charlin, and Chris J. Maddison. Learning to cut by looking ahead: Cutting plane selection via imitation learning. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), ´ *International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA*, volume 162 of *Proceedings of Machine Learning Research*, pp. 17584–17600. PMLR, 2022. URL <https://proceedings.mlr.press/v162/paulus22a.html>.
- <span id="page-14-5"></span>Pol Puigdemont, Stratis Skoulakis, Grigorios Chrysos, and Volkan Cevher. Learning to remove cuts in integer linear programming. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pp. 41235–41255. PMLR, 21–27 Jul 2024. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v235/puigdemont24a.html) [press/v235/puigdemont24a.html](https://proceedings.mlr.press/v235/puigdemont24a.html).
- <span id="page-14-3"></span>Lara Scavuzzo, Feng Chen, Didier Chetelat, Maxime Gasse, Andrea Lodi, Neil Yorke-Smith, and Karen Aardal. Learning to branch with tree mdps. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), *Advances in Neural Information Processing Systems*, volume 35, pp. 18514–18526. Curran Associates, Inc., 2022. URL [https://proceedings.neurips.cc/paper\\_files/paper/2022/](https://proceedings.neurips.cc/paper_files/paper/2022/file/756d74cd58592849c904421e3b2ec7a4-Paper-Conference.pdf) [file/756d74cd58592849c904421e3b2ec7a4-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/756d74cd58592849c904421e3b2ec7a4-Paper-Conference.pdf).
- <span id="page-14-2"></span>Lara Scavuzzo, Karen Aardal, Andrea Lodi, and Neil Yorke-Smith. Machine learning augmented branch and bound for mixed integer linear programming. *Mathematical Programming*, Aug 2024. ISSN 1436-4646. doi: 10.1007/s10107-024-02130-y. URL [https://doi.org/10.1007/](https://doi.org/10.1007/s10107-024-02130-y) [s10107-024-02130-y](https://doi.org/10.1007/s10107-024-02130-y).
- <span id="page-14-8"></span>Zhihao Shi, Xize Liang, and Jie Wang. LMC: Fast training of GNNs via subgraph sampling with provable convergence. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=5VBBA91N6n>.
- <span id="page-14-10"></span>Zhihao Shi, Jie Wang, Fanghua Lu, Hanzhu Chen, Defu Lian, Zheng Wang, Jieping Ye, and Feng Wu. Label deconvolution for node representation learning on large-scale attributed graphs against learning bias. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 46(12):11273– 11286, 2024. doi: 10.1109/TPAMI.2024.3459408.
- <span id="page-14-9"></span>Zhihao Shi, Jie Wang, Zhiwei Zhuang, Xize Liang, Bin Li, and Feng Wu. Accurate and scalable graph neural networks via message invariance. In *The Thirteenth International Conference on Learning Representations*, 2025. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=UqrFPhcmFp) [UqrFPhcmFp](https://openreview.net/forum?id=UqrFPhcmFp).
- <span id="page-14-6"></span>Jialin Song, ravi lanka, Yisong Yue, and Bistra Dilkina. A general large neighborhood search framework for solving integer linear programs. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 20012–20023. Curran Associates, Inc., 2020. URL [https://proceedings.neurips.cc/paper\\_files/paper/2020/](https://proceedings.neurips.cc/paper_files/paper/2020/file/e769e03a9d329b2e864b4bf4ff54ff39-Paper.pdf) [file/e769e03a9d329b2e864b4bf4ff54ff39-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/e769e03a9d329b2e864b4bf4ff54ff39-Paper.pdf).

- <span id="page-15-3"></span>Nicolas Sonnerat, Pengming Wang, Ira Ktena, Sergey Bartunov, and Vinod Nair. Learning a large neighborhood search algorithm for mixed integer programs. *ArXiv*, abs/2107.10201, 2021. URL <https://api.semanticscholar.org/CorpusID:236154746>.
- <span id="page-15-11"></span>Yongduo Sui, Wenyu Mao, Shuyao Wang, Xiang Wang, Jiancan Wu, Xiangnan He, and Tat-Seng Chua. Enhancing out-of-distribution generalization on graphs via causal attention learning. *ACM Trans. Knowl. Discov. Data*, 18(5), March 2024. ISSN 1556-4681. doi: 10.1145/3644392. URL <https://doi.org/10.1145/3644392>.
- <span id="page-15-0"></span>Yunhao Tang, Shipra Agrawal, and Yuri Faenza. Reinforcement learning for integer programming: Learning to cut. In Hal Daume III and Aarti Singh (eds.), ´ *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pp. 9367–9376. PMLR, 13–18 Jul 2020. URL [https://proceedings.mlr.press/v119/](https://proceedings.mlr.press/v119/tang20a.html) [tang20a.html](https://proceedings.mlr.press/v119/tang20a.html).
- <span id="page-15-12"></span>Haoyu Peter Wang, Jialin Liu, Xiaohan Chen, Xinshang Wang, Pan Li, and Wotao Yin. DIG-MILP: a deep instance generator for mixed-integer linear programming with feasibility guarantee. *Transactions on Machine Learning Research*, 2024a. ISSN 2835-8856. URL [https:](https://openreview.net/forum?id=MywlrEaFqR) [//openreview.net/forum?id=MywlrEaFqR](https://openreview.net/forum?id=MywlrEaFqR).
- <span id="page-15-10"></span>Jie Wang, Rui Yang, Zijie Geng, Zhihao Shi, Mingxuan Ye, Qi Zhou, Shuiwang Ji, Bin Li, Yongdong Zhang, and Feng Wu. Generalization in visual reinforcement learning with the reward sequence distribution. *CoRR*, abs/2302.09601, 2023a.
- <span id="page-15-2"></span>Jie Wang, Zhihai Wang, Xijun Li, Yufei Kuang, Zhihao Shi, Fangzhou Zhu, Mingxuan Yuan, Jia Zeng, Yongdong Zhang, and Feng Wu. Learning to cut via hierarchical sequence/set model for efficient mixed-integer programming. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, pp. 1–17, 2024b. doi: 10.1109/TPAMI.2024.3432716.
- <span id="page-15-9"></span>Zhihai Wang, Jie Wang, Qi Zhou, Bin Li, and Houqiang Li. Sample-efficient reinforcement learning via conservative model-based actor-critic. *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(8):8612–8620, Jun. 2022. doi: 10.1609/aaai.v36i8.20839. URL <https://ojs.aaai.org/index.php/AAAI/article/view/20839>.
- <span id="page-15-1"></span>Zhihai Wang, Xijun Li, Jie Wang, Yufei Kuang, Mingxuan Yuan, Jia Zeng, Yongdong Zhang, and Feng Wu. Learning cut selection for mixed-integer linear programming via hierarchical sequence model. In *The Eleventh International Conference on Learning Representations*, 2023b.
- <span id="page-15-8"></span>Zhihai Wang, Taoxing Pan, Qi Zhou, and Jie Wang. Efficient exploration in resource-restricted reinforcement learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(8): 10279–10287, Jun. 2023c. doi: 10.1609/aaai.v37i8.26224. URL [https://ojs.aaai.org/](https://ojs.aaai.org/index.php/AAAI/article/view/26224) [index.php/AAAI/article/view/26224](https://ojs.aaai.org/index.php/AAAI/article/view/26224).
- <span id="page-15-6"></span>Zhihai Wang, Lei Chen, Jie Wang, Yinqi Bai, Xing Li, Xijun Li, Mingxuan Yuan, Jianye Hao, Yongdong Zhang, and Feng Wu. A circuit domain generalization framework for efficient logic synthesis in chip design. In *Forty-first International Conference on Machine Learning*. PMLR, 2024c.
- <span id="page-15-5"></span>Zhihai Wang, Jie Wang, Qingyue Yang, Yinqi Bai, Xing Li, Lei Chen, Jianye HAO, Mingxuan Yuan, Bin Li, Yongdong Zhang, and Feng Wu. Towards next-generation logic synthesis: A scalable neural circuit generation framework. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024d. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=ZYNYhh3ocW) [ZYNYhh3ocW](https://openreview.net/forum?id=ZYNYhh3ocW).
- <span id="page-15-7"></span>Zhihai Wang, Jie Wang, Dongsheng Zuo, Yunjie Ji, Xinli Xia, Yuzhe Ma, Jianye Hao, Mingxuan Yuan, Yongdong Zhang, and Feng Wu. A hierarchical adaptive multi-task reinforcement learning framework for multiplier circuit design. In *Forty-first International Conference on Machine Learning*. PMLR, 2024e.
- <span id="page-15-4"></span>Zhihai Wang, Jie Wang, Xilin Xia, Dongsheng Zuo, Lei Chen, Yuzhe Ma, Jianye Hao, Mingxuan Yuan, and Feng Wu. Computing circuits optimization via model-based circuit genetic evolution. In *The Thirteenth International Conference on Learning Representations*, 2025.

- <span id="page-16-3"></span>Yaoxin Wu, Wen Song, Zhiguang Cao, and Jie Zhang. Learning large neighborhood search policy for integer programming. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), *Advances in Neural Information Processing Systems*, volume 34, pp. 30075–30087. Curran Associates, Inc., 2021. URL [https://proceedings.neurips.cc/paper\\_files/paper/2021/](https://proceedings.neurips.cc/paper_files/paper/2021/file/fc9e62695def29ccdb9eb3fed5b4c8c8-Paper.pdf) [file/fc9e62695def29ccdb9eb3fed5b4c8c8-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2021/file/fc9e62695def29ccdb9eb3fed5b4c8c8-Paper.pdf).
- <span id="page-16-8"></span>Rui Yang, Jie Wang, Zijie Geng, Mingxuan Ye, Shuiwang Ji, Bin Li, and Feng Wu. Learning task-relevant representations for generalization via characteristic functions of reward sequence distributions. In *The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 2242–2252. ACM, 2022.
- <span id="page-16-9"></span>Rui Yang, Jie Wang, Guoping Wu, and Bin Li. Uncertainty-based offline variational bayesian reinforcement learning for robustness under diverse data corruptions. In *Advances in Neural Information Processing Systems 38*, 2024.
- <span id="page-16-6"></span>Huigen Ye, Hua Xu, Hongyan Wang, Chengming Wang, and Yu Jiang. GNN&GBDT-guided fast optimizing framework for large-scale integer programming. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pp. 39864–39878. PMLR, 23–29 Jul 2023. URL <https://proceedings.mlr.press/v202/ye23e.html>.
- <span id="page-16-7"></span>Huigen Ye, Hua Xu, and Hongyan Wang. Light-MILPopt: Solving large-scale mixed integer linear programs with lightweight optimizer and small-scale training dataset. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=2oWRumm67L) [id=2oWRumm67L](https://openreview.net/forum?id=2oWRumm67L).
- <span id="page-16-0"></span>Taehyun Yoon. Confidence threshold neural diving. In *Advances in Neural Information Processing Systems, Competition Workshop on Machine Learning for Combinatorial Optimization*, 2021.
- <span id="page-16-1"></span>Giulia Zarpellon, Jason Jo, Andrea Lodi, and Yoshua Bengio. Parameterizing branch-and-bound search trees to learn branching policies. In *Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021*, pp. 3931–3939. AAAI Press, 2021. doi: 10.1609/AAAI. V35I5.16512. URL <https://doi.org/10.1609/aaai.v35i5.16512>.
- <span id="page-16-5"></span>Hao Zeng, Jiaqi Wang, Avirup Das, Junying He, Kunpeng Han, Haoyuan Hu, and Mingfei Sun. Effective generation of feasible solutions for integer programming via guided diffusion. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, KDD '24, pp. 4107–4118, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400704901. doi: 10.1145/3637528.3671783. URL [https://doi.org/10.1145/](https://doi.org/10.1145/3637528.3671783) [3637528.3671783](https://doi.org/10.1145/3637528.3671783).
- <span id="page-16-2"></span>Changwen Zhang, Wenli Ouyang, Hao Yuan, Liming Gong, Yong Sun, Ziao Guo, Zhichen Dong, and Junchi Yan. Towards imitation learning to branch for MIP: A hybrid reinforcement learning based sample augmentation approach. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=NdcQQ82mfy>.
- <span id="page-16-4"></span>Arman Zharmagambetov, Brandon Amos, Aaron Ferber, Taoan Huang, Bistra Dilkina, and Yuandong Tian. Landscape surrogate: Learning decision losses for mathematical optimization under partial information. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), *Advances in Neural Information Processing Systems*, volume 36, pp. 27332–27350. Curran Associates, Inc., 2023. URL [https://proceedings.neurips.cc/paper\\_files/paper/2023/file/](https://proceedings.neurips.cc/paper_files/paper/2023/file/574f145eac328cc4aaf9358e27120eb5-Paper-Conference.pdf) [574f145eac328cc4aaf9358e27120eb5-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/574f145eac328cc4aaf9358e27120eb5-Paper-Conference.pdf).

# A NOTATIONS

For a better understanding, we summarize and provide some key notations used in this paper in Table [4.](#page-17-1)

Table 4: Notations used in our paper.

<span id="page-17-1"></span>

| Notations    | Descriptions                                                                           |  |  |  |
|--------------|----------------------------------------------------------------------------------------|--|--|--|
| I            | A MILP instance.                                                                       |  |  |  |
| x            | The decision variables in the MILP.                                                    |  |  |  |
| xi           | th component of the solution decision variable.<br>The i                               |  |  |  |
| xˆ           | The predicted solution for the MILP.                                                   |  |  |  |
| x˜           | The reference solution for the MILP.                                                   |  |  |  |
| x[P]         | The partial solution the same variable values as x in the index set P.                 |  |  |  |
| Corr[P]<br>x | The corrected partial solution given the index set P.                                  |  |  |  |
| q(x   I)     | The distribution of the optimal solution given a MILP instance I.                      |  |  |  |
| pθ(x   I)    | The predicted marginal probabilities for the solution, given an instance I.            |  |  |  |
| q(x   xˆ, I) | The distribution of the reference solution given instance I and predicted solution xˆ. |  |  |  |
| DKL(·  ·)    | The KL divergence of two distributions.                                                |  |  |  |
| H(·)(·)      | The entropy of a distribution.                                                         |  |  |  |
| d(·, ·)      | The cross-entropy of two distributions.                                                |  |  |  |

# B THE SKETCH OF DERIVATION OF THE VARIABLE FIXING STRATEGY

### <span id="page-17-0"></span>B.1 THE MOTIVATION OF ASSUMPTION [1:](#page-6-4) FROM A CLASSICAL PROBLEM

In this part, We will provide some evidence to support Assumption [1](#page-6-4) from both an intuitive perspective for better understanding.

Example 1. Consider two students, A and B, participating in a math competition with a multiplechoice question offering two options. Let the probability that student A answers correctly be p, and the probability for student B be q. Since both have prepared for the exam, we assume p, q > 0.5. If they provide the same answer, what is the probability that their answer is correct? Conversely, if they provide different answers, what is the probability that student A is correct?

Answer. When both students provide the same answer, the probability that both are correct is given by:

P(A,B are correct | A,B provide the same answers)

$$= \frac{\mathbb{P}(A,B \text{ are correct and provide the same answers})}{\mathbb{P}(A,B \text{ provide the same answers})} = \frac{pq}{pq + (1-p)(1-q)}.$$
 (10)

Given that their answers are different, the probability that A is correct is

P(A is correct | A,B provide different answers)

$$= \frac{\mathbb{P}(\text{A is correct, and A,B provide different answers})}{\mathbb{P}(\text{A,B provide different answers})} = \frac{p(1-q)}{p(1-q) + (1-p)q}.$$
 (11)

A simple calculation reveals that

P(A,B are correct | A,B provide the same answers) > P(A is correct | A,B provide different answers), (12)

suggesting that if both students provide the same answer, it is more likely to be correct.

Returning to Assumption [1,](#page-6-4) we can draw an analogy: the predicted solution and the reference solution correspond to the answers given by the two students, while the optimal solution represents the correct answer. For simplicity, we treat these as independent. As shown in the example, ifs p, q > 0.5, then a consistent answer yields higher precision than differing answers. Given that the predictor is well-trained, we believe that the precision will exceed 0.5. Similarly, the precision of the traditional solver is also expected to be greater than 0.5.

$$q(x_{i} = 1 \mid \tilde{x}_{i} = 1, \hat{x}_{i} = 1, \mathcal{I}) \ge q(x_{i} = 1 \mid \tilde{x}_{i} = 0, \hat{x}_{i} = 1, \mathcal{I}), \text{ and}$$

$$q(x_{i} = 1 \mid \tilde{x}_{i} = 1, \hat{x}_{i} = 1, \mathcal{I}) \ge q(x_{i} = 1 \mid \tilde{x}_{i} = 1, \hat{x}_{i} = 0, \mathcal{I}).$$
(13)

#### B.2 THE SKETCH OF DERIVATION

In this paper, we propose the UEBO metric to determine the variables to fix. In practice, we do not calculate UEBO directly. Instead, we propose to approximate UEBO in Section [4.3.](#page-5-2)

• First, we introduce a concept called prediction-correction consistency (Definition [1](#page-6-5) in Section [4.3\)](#page-5-2).

$$-d(p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I}), q(\boldsymbol{x}_{i} \mid \hat{\boldsymbol{x}}_{i}, \mathcal{I})) = q(\boldsymbol{x}_{i} = 0 \mid \hat{\boldsymbol{x}}_{i}, \mathcal{I}) \log(1 - p_{\theta}(\boldsymbol{x}_{i} = 1 \mid \mathcal{I})) + q(\boldsymbol{x}_{i} = 1 \mid \hat{\boldsymbol{x}}_{i}, \mathcal{I}) \log(p_{\theta}(\boldsymbol{x}_{i} = 1 \mid \mathcal{I})),$$

$$(14)$$

which is the negative of cross entropy loss using reference solutions as 'labels'.

• Second, we also show that UEBO decreases as the prediction-correction consistency increases (Theorem [2](#page-6-1) in Section [4.3\)](#page-5-2). Using this property, to compare the UEBO of two variables, we just need to compare their prediction-correction consistency, with higher prediction-correction consistency indicating a lower UEBO. Here we provide a simple numerical example to explain this process.

<span id="page-18-0"></span>Example 2. We suppose k<sup>0</sup> = 0, k<sup>1</sup> = 3 and △ = 1 for simplicity. GNN prediction for an instance with five binary variables x<sup>i</sup> (i = 1, . . . , 5) is [0.9, 0.8, 0.7, 0.6, 0.5]. First, we add the following constraints to the instance,

$$x_{1} - 1 \leq \alpha_{1}, 1 - x_{1} \leq \alpha_{1},$$

$$x_{2} - 1 \leq \alpha_{2}, 1 - x_{2} \leq \alpha_{2},$$

$$x_{3} - 1 \leq \alpha_{3}, 1 - x_{3} \leq \alpha_{3},$$

$$\alpha_{1} + \alpha_{2} + \alpha_{3} \leq 1.$$
(15)

Suppose that the reference solution is [x˜1, x˜2, x˜3, x˜4, x˜5] = [1, 0, 1, 1, 1]. Thus, we have the prediction-correction consistency

$$-d(p_{\theta}(\mathbf{x}_{1} \mid \mathcal{I}), q(\mathbf{x}_{1} \mid \hat{\mathbf{x}}_{1}, \mathcal{I})) = 1 \times \log(0.9) + 0 \times \log(0.1) = \log(0.9),$$

$$-d(p_{\theta}(\mathbf{x}_{2} \mid \mathcal{I}), q(\mathbf{x}_{2} \mid \hat{\mathbf{x}}_{2}, \mathcal{I})) = 0 \times \log(0.8) + 1 \times \log(0.2) = \log(0.2),$$

$$-d(p_{\theta}(\mathbf{x}_{3} \mid \mathcal{I}), q(\mathbf{x}_{3} \mid \hat{\mathbf{x}}_{3}, \mathcal{I})) = 1 \times \log(0.7) + 0 \times \log(0.3) = \log(0.7).$$
(16)

Thus, the variables x<sup>1</sup> and x<sup>3</sup> have higher prediction-correction consistencies and thus lower UEBO. We have more confidence to fix x<sup>1</sup> and x3.

• Third, we step further to simplify the fixing rule. Intuitively, a well-trained model should satisfy the property that if the predicted partial solution xˆ<sup>i</sup> = 1, then pθ(x<sup>i</sup> | I) should be larger than 0.5; if the predicted partial solution xˆ<sup>i</sup> = 0, then pθ(x<sup>i</sup> | I) should be smaller than 0.5. Using this observation, we find the inequality

$$-d(p_{\theta}(\boldsymbol{x}_i \mid \mathcal{I}), q(\tilde{\boldsymbol{x}}_i \mid \hat{\boldsymbol{x}}_i, \mathcal{I})) \ge -d(p_{\theta}(\boldsymbol{x}_j \mid \mathcal{I}), q(\tilde{\boldsymbol{x}}_j \mid \hat{\boldsymbol{x}}_j, \mathcal{I}))$$
(17)

always hold given xˆ<sup>i</sup> = x˜<sup>i</sup> , xˆ<sup>j</sup> ̸= x˜<sup>j</sup> with xˆ<sup>i</sup> = xˆ<sup>j</sup> . To see this, assume that xˆ<sup>i</sup> = xˆ<sup>j</sup> = 1, we have pθ(x<sup>i</sup> | I) ≥ 0.5 and pθ(x<sup>j</sup> | I) ≥ 0.5. Then, we have

$$-d(p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I}), q(\tilde{\boldsymbol{x}}_{i} \mid \hat{\boldsymbol{x}}_{i}, \mathcal{I})) = \log(p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I})) \ge \log(0.5)$$
  
 
$$\ge \log(1 - p_{\theta}(\boldsymbol{x}_{j} \mid \mathcal{I})) = -d(p_{\theta}(\boldsymbol{x}_{j} \mid \mathcal{I}), q(\tilde{\boldsymbol{x}}_{j} \mid \hat{\boldsymbol{x}}_{j}, \mathcal{I})).$$
(18)

Thus, we propose the fixing strategy in Equation [\(7\)](#page-6-2). The proposed fixing strategy fixes variables satisfying xˆ<sup>i</sup> = x˜<sup>i</sup> , which are indeed the variables with high predictioncorrection consistency and thus low UEBO.

Example 3. In Example [2,](#page-18-0) xˆ<sup>1</sup> = x˜<sup>1</sup> = 1, xˆ<sup>2</sup> ̸= x˜2, xˆ<sup>3</sup> = x˜<sup>3</sup> = 1. Thus, we fix x<sup>1</sup> and x<sup>3</sup> to value 1. This result coincides with that given in Example [2.](#page-18-0)

### <span id="page-19-2"></span>C PROOF OF PROPOSITIONS AND THEOREMS.

#### C.1 PROOF OF PROPOSITION [1](#page-5-3)

We show the results by direct calculations,

$$D_{KL}(p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I})||q(\boldsymbol{x}_{i} \mid \mathcal{I})) = \int_{\mathcal{I}} \int_{\boldsymbol{x}_{i}} p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I}) p(\mathcal{I}) \log \left(\frac{p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I})}{q(\boldsymbol{x}_{i} \mid \mathcal{I})}\right) d\boldsymbol{x}_{i} d\mathcal{I}$$

$$= \int_{\mathcal{I}} \int_{\boldsymbol{x}_{i}} p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I}) p(\mathcal{I}) \log(p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I})) d\boldsymbol{x}_{i} d\mathcal{I} - \int_{\mathcal{I}} \int_{\boldsymbol{x}_{i}} p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I}) p(\mathcal{I}) \log(q(\boldsymbol{x}_{i} \mid \mathcal{I})) d\boldsymbol{x}_{i} d\mathcal{I}$$

$$= -\mathcal{H}(p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I})) - \int_{\mathcal{I}} \int_{\boldsymbol{x}_{i}} p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I}) p(\mathcal{I}) \log \left(\int_{\hat{\boldsymbol{x}}_{i}} q(\boldsymbol{x}_{i} \mid \hat{\boldsymbol{x}}_{i}, \mathcal{I}) p_{\theta}(\hat{\boldsymbol{x}}_{i} \mid \mathcal{I}) d\hat{\boldsymbol{x}}_{i}\right) d\boldsymbol{x}_{i} d\mathcal{I}$$

$$\leq -\mathcal{H}(p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I})) - \int_{\mathcal{I}} \int_{\boldsymbol{x}_{i}} \int_{\hat{\boldsymbol{x}}_{i}} p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I}) p(\mathcal{I}) p_{\theta}(\hat{\boldsymbol{x}}_{i} \mid \mathcal{I}) \log(q(\boldsymbol{x}_{i} \mid \hat{\boldsymbol{x}}_{i}, \mathcal{I})) d\hat{\boldsymbol{x}}_{i} d\boldsymbol{x}_{i} d\mathcal{I}$$

$$= -\mathcal{H}(p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I})) + d(p_{\theta}(\boldsymbol{x}_{i} \mid \mathcal{I}), q(\boldsymbol{x}_{i} \mid \hat{\boldsymbol{x}}_{i}, \mathcal{I})). \tag{19}$$

#### <span id="page-19-0"></span>C.2 PROOF OF THEOREM [2](#page-6-1)

We analyze the monotony property of UEBO as follows. First, we suppose that the variable x˜<sup>i</sup> takes value 1 in the reference solution, i.e., x˜<sup>i</sup> = 1. Thus, UEBO is in the form of

$$\begin{aligned}
&\text{UEBO}(p_{\theta}(\hat{\boldsymbol{x}}_{i} \mid \mathcal{I}), q(\tilde{\boldsymbol{x}}_{i} \mid \hat{\boldsymbol{x}}_{i}, \mathcal{I}))\Big|_{\tilde{\boldsymbol{x}}_{i}=1} = \mathcal{H}(p_{\theta}(\hat{\boldsymbol{x}}_{i} \mid \mathcal{I})) + d(p_{\theta}(\hat{\boldsymbol{x}}_{i} \mid \mathcal{I}), q(\tilde{\boldsymbol{x}}_{i} \mid \hat{\boldsymbol{x}}_{i}, \mathcal{I})) \\
&= -p_{\theta}(\hat{\boldsymbol{x}}_{i} \mid \mathcal{I}) \log p_{\theta}(\hat{\boldsymbol{x}}_{i} \mid \mathcal{I}) - (1 - p_{\theta}(\hat{\boldsymbol{x}}_{i} \mid \mathcal{I})) \log(1 - p_{\theta}(\hat{\boldsymbol{x}}_{i} \mid \mathcal{I})) - \log p_{\theta}(\hat{\boldsymbol{x}}_{i} \mid \mathcal{I}) \\
&= -(1 + p_{\theta}(\hat{\boldsymbol{x}}_{i} \mid \mathcal{I})) \log p_{\theta}(\hat{\boldsymbol{x}}_{i} \mid \mathcal{I}) - (1 - p_{\theta}(\hat{\boldsymbol{x}}_{i} \mid \mathcal{I})) \log(1 - p_{\theta}(\hat{\boldsymbol{x}}_{i} \mid \mathcal{I})).
\end{aligned} \tag{20}$$

Differentiating UEBO with respect to the predicted logit pθ(xˆ<sup>i</sup> | I), we have

$$\frac{d\text{UEBO}(p_{\theta}(\hat{\boldsymbol{x}}_i \mid \mathcal{I}) \Big|_{\hat{\boldsymbol{x}}_i = 1}}{dp_{\theta}(\hat{\boldsymbol{x}}_i \mid \mathcal{I})} = \log\left(\frac{1}{p_{\theta}(\hat{\boldsymbol{x}}_i \mid \mathcal{I})} - 1\right) - \frac{1}{p_{\theta}(\hat{\boldsymbol{x}}_i \mid \mathcal{I})} \le 0,\tag{21}$$

where pθ(xˆ<sup>i</sup> | I) takes values between [0, 1]. Therefore, UEBO decreases as pθ(xˆ<sup>i</sup> | I) becomes larger in [0, 1]. As pθ(xˆ<sup>i</sup> | I) grows, the cross-entropy term d(pθ(xˆ<sup>i</sup> | I), q(x˜<sup>i</sup> | xˆ<sup>i</sup> , I)) = − log pθ(xˆ<sup>i</sup> | I) decreases and the prediction-correction consistency becomes higher.

Similarly, the proof of case that the variable x˜<sup>i</sup> takes value 0 in the reference solution follows the same step, and we thus show that UEBO has a negative correlation with the prediction-correction consistency.

#### <span id="page-19-1"></span>C.3 PROOF OF THEOREM [3](#page-6-0)

We first expand the right-hand side of the inequalities.

$$q(\boldsymbol{x}_{i}^{*} = 1 \mid \tilde{\boldsymbol{x}}_{i} = 1, \mathcal{I})$$

$$=q(\boldsymbol{x}_{i}^{*} = 1 \mid \tilde{\boldsymbol{x}}_{i} = 1, \hat{\boldsymbol{x}}_{i} = 1, \mathcal{I})q(\hat{\boldsymbol{x}}_{i} = 1 \mid \tilde{\boldsymbol{x}}_{i} = 1, \mathcal{I})$$

$$+q(\boldsymbol{x}_{i}^{*} = 1 \mid \tilde{\boldsymbol{x}}_{i} = 1, \hat{\boldsymbol{x}}_{i} = 0, \mathcal{I})q(\hat{\boldsymbol{x}}_{i} = 0 \mid \tilde{\boldsymbol{x}}_{i} = 1, \mathcal{I})$$

$$\leq q(\boldsymbol{x}_{i}^{*} = 1 \mid \tilde{\boldsymbol{x}}_{i} = 1, \hat{\boldsymbol{x}}_{i} = 1, \mathcal{I})q(\hat{\boldsymbol{x}}_{i} = 1 \mid \tilde{\boldsymbol{x}}_{i} = 1, \mathcal{I})$$

$$+q(\boldsymbol{x}_{i}^{*} = 1 \mid \tilde{\boldsymbol{x}}_{i} = 1, \hat{\boldsymbol{x}}_{i} = 1, \mathcal{I})q(\hat{\boldsymbol{x}}_{i} = 0 \mid \tilde{\boldsymbol{x}}_{i} = 1, \mathcal{I}),$$

$$(22)$$

where the inequality holds by the consistency conditions [\(8\)](#page-6-3). Thus we have

$$q(\boldsymbol{x}_{i}^{*} = 1 \mid \tilde{\boldsymbol{x}}_{i} = 1, \mathcal{I})$$

$$\leq q(\boldsymbol{x}_{i}^{*} = 1 \mid \tilde{\boldsymbol{x}}_{i} = 1, \hat{\boldsymbol{x}}_{i} = 1, \mathcal{I}) (q(\hat{\boldsymbol{x}}_{i} = 1 \mid \tilde{\boldsymbol{x}}_{i} = 1, \mathcal{I}) + q(\hat{\boldsymbol{x}}_{i} = 0 \mid \tilde{\boldsymbol{x}}_{i} = 1, \mathcal{I}))$$

$$= q(\boldsymbol{x}_{i}^{*} = 1 \mid \tilde{\boldsymbol{x}}_{i} = 1, \hat{\boldsymbol{x}}_{i} = 1, \mathcal{I}).$$
(23)

### C.4 PROOF OF FEASIBILITY GUARANTEE

Since the trust-region searching problem [2](#page-3-1) is feasible, we can obtain a feasible solution as a reference solution, denoted by x˜. Thus, x˜ satisfies the constraint in Problem [2,](#page-3-1) i.e.,

$$A\tilde{x} \leq b$$
,  $l \leq \tilde{x} \leq u$ ,  $\tilde{x}[P] \in \mathcal{B}_P(\hat{x}[P], \triangle)$ ,  $\tilde{x} \in \mathbb{Z}^p \times \mathbb{R}^{n-p}$ .

According to the consistency-based variable fixing strategy, we fix the variable x[P ′ ] in the instance I to values x˜[P ′ ], where P ′ is a subset of P satisfying x˜[P ′ ] = xˆ[P ′ ]. Therefore, we have

$$\bm{A}\tilde{\bm{x}} \leq \bm{b}, \quad \bm{l} \leq \tilde{\bm{x}} \leq \bm{u}, \quad \tilde{\bm{x}}[P'] = \bm{x}^{Corr}[P'], \quad \tilde{\bm{x}} \in \mathbb{Z}^p \times \mathbb{R}^{n-p}.$$

This implies that x˜ is also a feasible solution of Problem [6.](#page-5-1)

# <span id="page-20-1"></span>D IMPLEMENTATION OF OUR METHODS AND THE BASELINES

Machine learning has made great progress in a broad range of areas [\(Mao et al., 2025a;](#page-13-12)[b;](#page-14-7) [Dong](#page-11-10) [et al., 2025;](#page-11-10) [Bai et al., 2025;](#page-10-5) [Wang et al., 2025;](#page-15-4) [2024d](#page-15-5)[;c;](#page-15-6)[e;](#page-15-7) [2023c;](#page-15-8) [2022;](#page-15-9) [Yang et al., 2022;](#page-16-8) [Liu](#page-13-13) [et al., 2023b;](#page-13-13) [Wang et al., 2023a;](#page-15-10) [Yang et al., 2024\)](#page-16-9) and graph neural networks have shown great performance in solving MILPs. The PS models used in this paper align with those outlined in the original papers [Han et al.](#page-12-0) [\(2023\)](#page-12-0). We use the code in [https://github.com/sribdcn/](https://github.com/sribdcn/Predict-and-Search) [Predict-and-Search](https://github.com/sribdcn/Predict-and-Search) MILP method to implement PS. For the PS predictor, we leverage a graph neural network [\(Kipf & Welling, 2017;](#page-12-12) [Shi et al., 2023;](#page-14-8) [Sui et al., 2024;](#page-15-11) [Mao et al., 2024;](#page-13-14) [Shi et al., 2025;](#page-14-9) [2024\)](#page-14-10) comprising four half-convolution layers. The codes of the original work of ND [\(Nair et al., 2020\)](#page-14-0) and ConPS [\(Huang et al., 2024\)](#page-12-1) are not publicly available. We try our best to reproduce these baselines and tune the hyperparameters for testing. We conducted all the experiments on a single machine with NVidia GeForce GTX 3090 GPUs and Intel(R) Xeon(R) E5-2667 V4CPUs 3.20GHz.

In the training process of the predictors, we set the initial learning rate to be 0.001 and the training epoch to be 10,000 with early stopping. To collect the training data, we run a single thread Gurobi on each training and validation instance to for 3,600 seconds and record the best 50 solutions.

The partial solution size parameter (k0, k1) and neighborhood parameter ∆ are two important parameters in PS. The partial solution size parameter (k0, k1, ∆) represents the numbers of variables fixed with values 0 and 1 in a partial solution. The neighborhood parameter ∆ defines the radius of the searching neighborhood. We list these two parameters used in our experiments in Table [5.](#page-20-2)

<span id="page-20-2"></span>Table 5: The partial solution size parameter (k0, k1) and neighborhood parameter ∆.

| Benchmark               | CA                       | SC                           | IP                     | WA                      |
|-------------------------|--------------------------|------------------------------|------------------------|-------------------------|
| PS+Gurobi               | (600,0,1)                | (2000,0,100)                 | (400,5,10)             | (0,500,10)              |
| ConPS+Gurobi<br>PS+SCIP | (900,0,50)<br>(400,0,10) | (1000,0,200)<br>(2000,0,100) | (400,5,3)<br>(400,5,1) | (0,500,10)<br>(0,600,5) |
| ConPS+SCIP              | (900,0,50)               | (1000,0,200)                 | (400,5,3)              | (0,400,50)              |

For our Apollo-MILP, the training process follows a similar approach, but we incorporate data augmentation to align with the testing distributions. For each original training instance with n variables, we select the best solution x ∗ from the solution pool S. Then, we randomly sample a fixing ratio α ∈ [0.3, 0.7] and an index set P<sup>α</sup> of variables, where the index set contains αn elements. Finally, we enforce the variables in the set V<sup>α</sup> to the corresponding values in x ∗ , resulting in x[Pα] = x ∗ [Pα]. By varying the ratio α, we generate five reduced augmented instances from each training instance.

# <span id="page-20-0"></span>E DETAILS ON BIPARTITE GRAPH REPRESENTATIONS

The bipartite instance graph representation utilized in this paper closely aligns with that presented in the PS paper [Han et al.](#page-12-0) [\(2023\)](#page-12-0). We list the graph features in Table [6.](#page-21-1)

<span id="page-21-1"></span>Table 6: The variable features, constraint features, and edge features used for the predictor.

| Index | Variable Feature Name        | Description                                                                          |
|-------|------------------------------|--------------------------------------------------------------------------------------|
| 0     | Objective                    | Normalized objective coefficient                                                     |
| 1     | Variable coefficient         | Average variable coefficient in all constraints                                      |
| 2     | Variable degree              | Degree of the variable node in the bipartite graph<br>representation                 |
| 3     | Maximum variable coefficient | Maximum variable coefficient in all constraints                                      |
| 4     | Minimum variable coefficient | Minimum variable coefficient in all constraints                                      |
| 5     | Variable type                | Whether the variable is an integer variable or not)                                  |
| 6-17  | Position embedding           | Binary encoding of the order of appearance for<br>each variable among all variables. |
| Index | Constraint Feature Name      | Description                                                                          |
| 0     | Constraint coefficient       | Average of all coefficients in the constraint                                        |
| 1     | Constraint degree            | Degree of constraint nodes                                                           |
| 2     | Bias                         | Normalized right-hand-side of the constraint                                         |
| 3     | Sense                        | The sense of the constraint                                                          |
| Index | Constraint Feature Name      | Description                                                                          |
| 0     | Coefficient                  | Constraint coefficient                                                               |

# <span id="page-21-0"></span>F DETAILS ON THE BENCHMARKS

#### F.1 BENCHMARKS IN MAIN EVALUATION

The CA and SC benchmark instances are generated following the process described in [Gasse et al.](#page-11-1) [\(2019\)](#page-11-1). Specifically, the CA instances were generated using the algorithm from [Leyton-Brown et al.](#page-13-11) [\(2000\)](#page-13-11), and the SC instances were generated using the algorithm presented in [Balas & Ho](#page-10-4) [\(1980\)](#page-10-4). The IP and WA instances are obtained from the NeurIPS ML4CO 2021 competition [\(Gasse et al.,](#page-11-0) [2022\)](#page-11-0). The statistical information for all the instances is provided in Table [7.](#page-21-2)

<span id="page-21-2"></span>Table 7: Statistical information of the benchmarks we used in this paper.

|                                | CA   | SC   | IP   | WA    |
|--------------------------------|------|------|------|-------|
| Constraint Number              | 2593 | 3000 | 195  | 64306 |
| Variable Number                | 1500 | 5000 | 1083 | 61000 |
| Number of Binary Variables     | 1500 | 5000 | 1050 | 1000  |
| Number of Continuous Variables | 0    | 0    | 33   | 60000 |
| Number of Integer Variables    | 0    | 0    | 0    | 0     |

#### <span id="page-21-4"></span>F.2 BENCHMARKS IN USED FOR GENERALIZATION

We generate larger CA and SC instances to evaluate the generalization ability of the approaches. We use the code in [Gasse et al.](#page-11-1) [\(2019\)](#page-11-1) for data generation. Specifically, the generated CA instances have an average of 2,596 constraints and 4,000 variables, and the SC instances have 6,000 constraints and 10,000 variables. These instances are considerably larger than the training instances.

#### <span id="page-21-3"></span>F.3 SUBSET OF MIPLIB

We construct a subset of MIPLIB [\(Gleixner et al., 2021\)](#page-11-11) to evaluate the solvers' ability to handle challenging real-world instances. Specifically, we select instances based on their similarity, which is measured by 100 human-designed features [\(Gleixner et al., 2021\)](#page-11-11). Instances with presolving times exceeding 300 seconds or those that exceed GPU memory limits during the inference process are discarded. Inspired by the IIS dataset used in [Wang et al.](#page-15-12) [\(2024a\)](#page-15-12), we develop a refined IIS dataset containing eleven instances. We divide this dataset into training and testing sets, comprising eight training instances and three testing instances (ramos3, scpj4scip, and scpl4). Detailed information on the IIS dataset can be found in Table [8.](#page-22-1)

Table 8: Statistical information of the instances in the constructed IIS dataset.

<span id="page-22-1"></span>

| Instance Name   | Constraint Number | Variable Number | Nonzero Coefficient Number |
|-----------------|-------------------|-----------------|----------------------------|
| ex1010-pi       | 1468              | 25200           | 102114                     |
| fast0507        | 507               | 63009           | 409349                     |
| glass-sc        | 6119              | 214             | 63918                      |
| iis-glass-cov   | 5375              | 214             | 56133                      |
| iis-hc-cov      | 9727              | 297             | 142971                     |
| ramos3          | 2187              | 2187            | 32805                      |
| scpj4scip       | 1000              | 99947           | 999893                     |
| scpk4           | 2000              | 100000          | 1000000                    |
| scpl4           | 2000              | 200000          | 2000000                    |
| seymour         | 4944              | 1372            | 33549                      |
| v150d30-2hopcds | 7822              | 150             | 103991                     |

#### F.4 MORE BENCHMARKS

To demonstrate the effectiveness of our method, we include three more benchmarks. The statistical information is in Table [9.](#page-22-2)

SCUC dataset This dataset comes from the Energy Electronics Industry Innovation Competition. The benchmark contains large-scale instances from real-world power systems.

Smaller-size CA dataset This dataset is the CA dataset with the same sizes as the 'Hard CA' in [Gasse et al.](#page-11-1) [\(2019\)](#page-11-1). This dataset has a lower instance size and computational hardness than the CA dataset used in our paper. All the methods can solve the problems within the time limit, and we thus report the solving time.

<span id="page-22-2"></span>APS dataset This dataset is from an anonymous commercial enterprise containing real-world production scheduling problems in the factory. The instances in APS are general integer programming problems, containing binary, continuous, and general integer variables.

Table 9: Statistical information of the benchmarks.

|                                | SCUC  | Small-size CA | APS   |
|--------------------------------|-------|---------------|-------|
| Constraint Number              | 27835 | 576           | 31296 |
| Variable Number                | 19807 | 1500          | 31344 |
| Number of Binary Variables     | 9295  | 1500          | 1500  |
| Number of Continuous Variables | 10512 | 0             | 15984 |
| Number of Integer Variables    | 0     | 0             | 9600  |

### <span id="page-22-0"></span>G HYPERPARAMETERS

We report the hyperparameters (k (i) 0 , k(i) 1 , △(i) ) of Apollo-MILP used in Table [10.](#page-23-1)

| Table 10: Hyperparameters (k | (i)<br>, k(i)<br>, △(i)<br>0<br>1 | ) for Apollo-MILP. |
|------------------------------|-----------------------------------|--------------------|
|------------------------------|-----------------------------------|--------------------|

<span id="page-23-1"></span>

|             | CA         | SC           | IP          | WA           |
|-------------|------------|--------------|-------------|--------------|
| Iteration 1 | (400,0,60) | (1000,0,200) | (100,20,50) | (20,200,100) |
| Iteration 2 | (200,0,30) | (500,0,100)  | (40,15,20)  | (10,100,50)  |
| Iteration 3 | (100,0,15) | (250,0,50)   | (20,15,10)  | (10,5,5)     |
| Iteration 4 | (50,0,10)  | (10,0,5)     | (5,50,30)   | (1,10,5)     |

# H ADDITIONAL EXPERIMENT RESULTS

#### <span id="page-23-0"></span>H.1 REAL-WORLD DATASET

To further demonstrate the applicability of Apollo-MILP, we conduct experiments on instances from MIPLIB [\(Gleixner et al., 2021\)](#page-11-11), a challenging real-world dataset. Due to the heterogeneous nature of the instances in MIPLIB, applying ML-based solvers directly to the entire dataset can be difficult. However, we can focus on a subset of MIPLIB that contains similar instances [\(Wang et al., 2023b;](#page-15-1) [2024a\)](#page-15-12). For more information on the selected MILP subset, referred to as IIS, please see Appendix [F.3.](#page-21-3) This subset consists of eleven challenging real-world instances. Notice that we need to carefully tune the hyperparameters (k0, k1, △) for the ML baselines, as improper hyperparameters can easily lead to infeasibility. While Appolo-MILP exhibits a strong adaptation across different hyperparameters. We report the solving performance of the solvers in Table [11,](#page-23-2) where Apollo-MILP significantly outperforms other baselines, showcasing its promising potential for real-world applications. We also report the detailed results of the real-world challenging MIPLIB dataset mentioned. We set the time limit to 3,600 seconds and ran two iterations with 1,000 and 2,600 seconds. Given the variation in instance sizes within the dataset, we set △ = 1000 and the proportion of fixed variables to (α0, α1) = (0.8, 0), which means that we fix 0.8 of the binary variables to 0 in the first round.

<span id="page-23-2"></span>Table 11: The results in the IIS dataset, which is used in [Wang et al.](#page-15-12) [\(2024a\)](#page-15-12) and is a subset of MIPLIB. We build the ML approaches on Gurobi and set the solving time limit to 3,600s.

|        | Obj ↓  | gapabs<br>↓ |
|--------|--------|-------------|
| Gurobi | 214.00 | 23.00       |
| ND     | 213.00 | 22.00       |
| PS     | 211.00 | 20.00       |
| ConPS  | 211.00 | 20.00       |
| Ours   | 209.33 | 18.33       |
|        |        |             |

Table 12: The best objectives found by the approaches on each test instance in IIS. *BKS* represents the best objectives from the website of MIPLIB <https://miplib.zib.de/index.html>.

|           | BKS    | Gurobi | ND     | PS     | ConPS  | Ours   |
|-----------|--------|--------|--------|--------|--------|--------|
| ramos3    | 186.00 | 233.00 | 233.00 | 225.00 | 225.00 | 224.00 |
| scpj4scip | 128.00 | 132.00 | 131.00 | 133.00 | 133.00 | 131.00 |
| scpl4     | 259.00 | 277.00 | 275.00 | 275.00 | 275.00 | 273.00 |

#### H.2 HYPERPARAMETER ANALYSIS

We investigate the impact of hyperparameters in our proposed framework. In this part, we conduct extensive experiments to analyze the impact of hyperparameters, including the partial solution size, neighborhood parameter, iteration time, iteration round number, and the use of data augmentation. For all experiments, we implement the method using Gurobi and set a time limit of 1,000 seconds.

Partial Solution Size Parameters We examine the effects of the partial solution size parameter kfix in the CA benchmark. As noted in [Huang et al.](#page-12-1) [\(2024\)](#page-12-1), fixing k (i) <sup>1</sup> = 0 always yields better solutions. Therefore, we focus on the effects of k (i) <sup>0</sup> while keeping k (i) <sup>1</sup> = 0 and △ = (60, 30, 15, 0) for four rounds of iterations. Our findings, presented in Table [13,](#page-24-0) indicate that a fixing coverage of 50% of variables yields the best performance. This optimal coverage balances the risk of the solution prediction methods becoming trapped in low-quality neighborhoods—–common with high coverage of fixed variables——while avoiding ineffective problem reduction associated with low coverage.

<span id="page-24-0"></span>Table 13: The solving performance with different partial solution size parameters kfix on the CA benchmark, under the time limit of 1,000 seconds. The coverage rate implies the approximate proportion of fixing variables.

|              | Obj ↑    | ↓<br>gapabs |
|--------------|----------|-------------|
| Coverage 85% | 96950.55 | 666.04      |
| Coverage 70% | 96929.98 | 686.61      |
| Coverage 50% | 97487.18 | 129.41      |
| Coverage 30% | 97359.09 | 257.50      |
|              |          |             |

Neighborhood Parameter We examine how the choice of neighborhood parameters affects solving performance. From Table [14,](#page-24-1) we can see that when △ is small, increasing its value can enhance performance since searching within a larger area may yield higher-quality solutions. However, when △ is too large, performance decreases, as the expanded trust region results in a larger search space, leading to inefficiencies in the search process.

<span id="page-24-1"></span>Table 14: The effects of neighborhood parameters on the solving performance.

|                       | Obj ↑    | ↓<br>gapabs |
|-----------------------|----------|-------------|
| 60% of Fixing Numbers | 97297.52 | 319.07      |
| 50% of Fixing Numbers | 97343.47 | 273.12      |
| 20% of Fixing Numbers | 97487.18 | 129.41      |
| 10% of Fixing Numbers | 97019.17 | 597.12      |

Iteration Time We investigate the relationship between iteration time and performance in Table [15,](#page-24-2) setting a time limit of 1,000 seconds. The early iterations focus on identifying a high-quality feasible solution to reduce the search space, while the final iteration aims to exploit the optimal solution within this reduced space. Through four iterations, we find that the last iteration, allocated 600 seconds, yields the best performance. As the exploitation time increases, the algorithm is more likely to search the reduced space thoroughly. However, extending the exploitation time reduces the available time to enhance the predicted solution during the early stages, which can lead to a decline in the quality of the predicted solutions.

<span id="page-24-2"></span>Table 15: The effects of iteration time on the solving performance.

|                   | Obj ↑    | ↓<br>gapabs |
|-------------------|----------|-------------|
| (25,25,50,900)    | 96741.04 | 875.55      |
| (50,50,100,800)   | 96889.70 | 726.89      |
| (100,100,200,600) | 97487.18 | 129.41      |
| (150,150,300,400) | 97353.35 | 263.24      |

The Rounds of Iterations We conduct experiments on the rounds of iterations. We fix the solving time limit to 1,000s and compare different rounds of iterations in the CA dataset. Given an iteration round, we set the same search time across the iterations. The results are presented in Table [16,](#page-25-1) which indicates the four rounds of iterations have the best performance.

Data Augmentation The distribution of the reduced problems in each iteration may differ from that of the original problems. As pointed out in Section [4.1,](#page-3-0) we employ data augmentation to align the distributional shifts. We conduct experiments to evaluate the effect of data augmentation in

<span id="page-25-1"></span>Table 16: The effects of rounds of the iterations on solving performance.

|                       | Obj ↑    | ↓<br>gapabs |
|-----------------------|----------|-------------|
| (500,500)             | 97132.39 | 484.20      |
| (333.3,333.3,333.3)   | 97349.74 | 266.85      |
| (250,250,250,250)     | 97388.21 | 228.38      |
| (200,200,200,200,200) | 96889.70 | 726.89      |
|                       |          |             |

<span id="page-25-2"></span>Table [17.](#page-25-2) We use the CA dataset and set the solving time limit to 1,000s. The results show that data augmentation can improve the performance.

Table 17: The effects of data augmentation on solving performance.

|                       | Obj ↑    | ↓<br>gapabs |
|-----------------------|----------|-------------|
| w/o data augmentation | 97393.65 | 222.94      |
| Ours                  | 97487.18 | 129.41      |

#### <span id="page-25-0"></span>H.3 COMPARISON WITH MORE BASELINES

Comparison with Different Fixing Strategies We provide more results when comparing different fixing strategies on more benchmarks. We set the search time in each iteration to be consistent across these methods. Direct Fixing relies totally on the GNN predictor for variable fixing, and Multi-stage PS relies on the reference solution provided by the traditional solver. Different from these two baselines, our proposed method introduces the correction mechanism and combines the predicted and reference solutions, determining the most confident variables to fix. The results are presented in Table [18.](#page-25-3) The results in Table [18](#page-25-3) show that our proposed prediction-correction method outperforms the other baselines.

<span id="page-25-3"></span>Table 18: Comparison of solving performance between our approach and different fixing strategies, under a 1, 000s time limit. We report the average best objective values and absolute primal gap.

|                       | CA (BKS 97616.59) |             | SC (BKS 122.95) |             | IP (BKS 8.90) |             | WA (BKS 704.88) |             |
|-----------------------|-------------------|-------------|-----------------|-------------|---------------|-------------|-----------------|-------------|
|                       | Obj ↑             | ↓<br>gapabs | Obj ↓           | ↓<br>gapabs | Obj ↓         | ↓<br>gapabs | Obj ↓           | ↓<br>gapabs |
| Gurobi                | 97297.52          | 319.07      | 123.40          | 0.45        | 9.38          | 0.48        | 705.49          | 0.61        |
| PS+Gurobi             | 97358.23          | 258.36      | 123.30          | 0.34        | 9.17          | 0.27        | 705.45          | 0.57        |
| Direct Fixing+Gurobi  | 96939.19          | 677.4       | 123.30          | 0.35        | 9.22          | 0.32        | 705.40          | 0.52        |
| Multi-stage PS+Gurobi | 97016.47          | 600.12      | 123.20          | 0.25        | 9.18          | 0.28        | 705.33          | 0.45        |
| Ours+Gurobi           | 97487.18          | 129.41      | 123.05          | 0.10        | 8.90          | 0.00        | 704.88          | 0.00        |

Comparison with a Warm-starting Gurobi We provide more results when comparing the warmstarting methods on more benchmarks. The results are presented in Table [19,](#page-26-1) in which we set the solving time limit as 1,000s.

#### H.4 EVALUATION ON MORE BENCHMARKS

We evaluate our method in more benchmarks, including the small-scale CA dataset, the large-scale and real-world dataset SCUC, and real-world general integer programming problems APS. We report the experiment results in Table [20.](#page-26-2) The experiment results demonstrate the strong performance of our proposed method across various benchmarks. Notice that our method still outperforms the baselines in the general integer programming problems (APS) in Table [20.](#page-26-2)

<span id="page-26-1"></span>Table 19: Comparison of solving performance between our approach and the warm-starting methods, under a 1, 000s time limit. We report the average best objective values and absolute primal gap.

|                | CA (BKS 97616.59) |             | SC (BKS 122.95) |             | IP (BKS 8.90) |             | WA (BKS 704.88) |             |
|----------------|-------------------|-------------|-----------------|-------------|---------------|-------------|-----------------|-------------|
|                | Obj ↑             | ↓<br>gapabs | Obj ↓           | ↓<br>gapabs | Obj ↓         | ↓<br>gapabs | Obj ↓           | ↓<br>gapabs |
| Gurobi         | 97297.52          | 319.07      | 123.40          | 0.45        | 9.38          | 0.48        | 705.49          | 0.61        |
| PS+Gurobi      | 97358.23          | 258.36      | 123.30          | 0.35        | 9.17          | 0.27        | 705.45          | 0.57        |
| WS-PS+Gurobi   | 97016.34          | 600.25      | 123.30          | 0.35        | 9.20          | 0.30        | 705.45          | 0.57        |
| WS-Ours+Gurobi | 97359.19          | 257.40      | 123.20          | 0.25        | 9.13          | 0.23        | 705.40          | 0.52        |
| Ours+Gurobi    | 97487.18          | 129.41      | 123.05          | 0.10        | 8.90          | 0.00        | 704.88          | 0.00        |

<span id="page-26-2"></span>Table 20: Comparison of solving performance on more benchmarks, under a 1, 000s time limit. We report the average best objective values and absolute primal gap.

|              | SCUC (BKS 1254399.66) |             | Smaller-Size CA |             | APS (BKS 558917.52) |             |  |
|--------------|-----------------------|-------------|-----------------|-------------|---------------------|-------------|--|
|              | Obj ↓                 | gapabs<br>↓ | Time ↓          | gapabs<br>↓ | Obj ↓               | gapabs<br>↓ |  |
| Gurobi       | 1269353.86            | 14954.20    | 105.61          | 0.00        | 666583.20           | 107665.68   |  |
| ND+Gurobi    | 1266355.77            | 11956.11    | 102.34          | 0.00        | 646908.07           | 87990.55    |  |
| PS+Gurobi    | 1265332.21            | 10932.55    | 104.68          | 0.00        | 635087.43           | 76169.91    |  |
| ConPS+Gurobi | 1264173.98            | 9774.32     | 98.63           | 0.00        | 626713.56           | 67796.04    |  |
| Ours+Gurobi  | 1261684.89            | 7285.23     | 94.04           | 0.00        | 603443.23           | 44525.71    |  |

### <span id="page-26-0"></span>H.5 GENERALIZATION

We evaluate the generalization performance of our method. Specifically, we generate larger instances of the CA and SC problems (please refer to Appendix [F.2](#page-21-4) for more details). We utilize the model trained on the instances described in Section [5.1](#page-7-1) and evaluate the models on these larger instances. The experiment results in Table [21](#page-26-3) demonstrate the strong generalization ability of Apollo-MILP, as it outperforms other baselines on these larger instances.

<span id="page-26-3"></span>Table 21: We evaluate the generalization performance on 100 larger instances. The ML approaches are implemented using Gurobi, with a time limit set to 1,000s. '↑' indicates that higher is better, and '↓' indicates that lower is better. We mark the best values in bold.

|              | CA (BKS 115746.88) |             | SC (BKS 101.45) |             |  |
|--------------|--------------------|-------------|-----------------|-------------|--|
|              | Obj ↑              | ↓<br>gapabs | Obj ↓           | ↓<br>gapabs |  |
| Gurobi       | 114960.25          | 786.63      | 102.29          | 0.84        |  |
| ND+Gurobi    | 115035.44          | 711.44      | 102.51          | 1.06        |  |
| PS+Gurobi    | 115228.20          | 518.68      | 102.27          | 0.82        |  |
| ConPS+Gurobi | 115343.23          | 403.65      | 102.18          | 0.73        |  |
| Ours+Gurobi  | 115413.86          | 333.02      | 102.16          | 0.71        |  |

# I REPRODUCTION OF THE BASELINES

Since ND and ConPS are not open-source, we must reproduce the results from the original papers to validate our models. In this section, we reproduce the experiments conducted in the original studies [\(Nair et al., 2020\)](#page-14-0) and [\(Huang et al., 2024\)](#page-12-1) to ensure the performance of our reproduced models.

Following the validation approach and settings outlined in [Nair et al.](#page-14-0) [\(2020\)](#page-14-0) and [Han et al.](#page-12-0) [\(2023\)](#page-12-0), we conduct experiments on the Neural Network Verification (NNV) dataset. We implemented the Neural Diving method within SCIP and compared its performance against the default SCIP. As noted by [Han et al.](#page-12-0) [\(2023\)](#page-12-0), tuning the presolve option in SCIP can lead to false assertions of feasibility in the NNV dataset; therefore, we disabled this option for both SCIP and our reproduced ND+SCIP. The results are presented in Figure [4,](#page-27-0) where ND significantly outperforms SCIP, confirming the performance of our reproduced model.

<span id="page-27-0"></span>To reproduce ConPS, we utilized the IP dataset built on SCIP and replicated the experiments described in the original paper [\(Huang et al., 2024\)](#page-12-1). The results are summarized in Figure [5,](#page-27-1) where the performance of ConPS aligns with the results in the original study [\(Huang et al., 2024\)](#page-12-1).

<span id="page-27-1"></span>Figure 4: The reproduced results of SCIP and ND+SCIP on the NNV dataset.

Figure 5: The reproduced results of SCIP and ConPS+SCIP on the IP dataset.