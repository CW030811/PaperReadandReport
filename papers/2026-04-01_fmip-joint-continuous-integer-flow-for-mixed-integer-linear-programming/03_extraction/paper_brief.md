# Paper Brief

## Paper Metadata
- Title: FMIP: Joint Continuous-Integer Flow For Mixed-Integer Linear Programming
- Authors: Hongpei Li, Hui Yuan, Han Zhang, Jianghao Lin, Dongdong Ge, Mengdi Wang, Yinyu Ye
- Venue / Year: ICLR 2026
- Paper Folder: papers/2026-04-01_fmip-joint-continuous-integer-flow-for-mixed-integer-linear-programming

## 1. Research Question
- Summary: Can a generative MILP heuristic model the joint distribution of both integer and continuous variables, instead of only integer variables, to produce better warm starts and solver guidance?
- Evidence: Abstract; Section 1 Introduction

## 2. High-Level Algorithmic Idea
- Summary: FMIP uses a joint continuous-integer generative framework based on conditional flow matching to progressively generate full MILP solutions. Because it jointly models all variables, it can apply holistic guidance from objective values and constraint violations during inference.
- Evidence: Abstract; Section 1 Introduction

## 3. Training Label / Dataset
- Summary: The model learns from MILP solution distributions and is evaluated on eight standard MILP benchmarks. The quick scan does not recover the exact benchmark family names, but the paper explicitly emphasizes broad compatibility across multiple backbones and downstream solvers.
- Evidence: Abstract; Section 1 Introduction

## 4. Network Structure
- Summary: The framework is generative rather than purely discriminative. It is built on a joint continuous-integer flow formulation and is designed to be backbone-agnostic, so the core architectural novelty lies in the generative modeling and guidance paradigm rather than one fixed encoder.
- Evidence: Abstract; Section 1 Introduction

## 5. Network Output
- Summary: The model outputs a complete MILP solution sample containing both integer and continuous decision variables.
- Evidence: Abstract; Section 1 Introduction

## 6. Loss Function
- Summary: The paper explicitly states that FMIP leverages conditional flow matching. The exact full loss decomposition is not fully extracted in this quick pass, but flow-matching-based training is a core part of the method.
- Evidence: Section 1 Introduction

## 7. Inference Policy
- Summary: During inference, FMIP iteratively refines a generated solution trajectory while using holistic guidance from objective feedback and constraint violations to steer samples toward better feasibility and optimality before handing them to downstream solvers.
- Evidence: Abstract; Section 1 Introduction

## 8. Novel Contribution
- Summary: The main novelty is being the first generative MILP framework to jointly model integer and continuous variables, thereby removing an information bottleneck in prior MILP generators and enabling richer guidance at inference time.
- Evidence: Abstract; Section 1 Introduction

## Evidence Pointers
- Abstract: states the joint continuous-integer idea and empirical gains
- Section 1: explains the information bottleneck of integer-only modeling
- Figure 1: illustrates joint modeling and holistic guidance
- arXiv 2507.23390 / ICLR 2026 version front matter: authors and venue status

## Unresolved Questions
- The exact eight benchmark families should be confirmed from the experiments section.
- The exact objective terms beyond high-level conditional flow matching need a deeper read.
