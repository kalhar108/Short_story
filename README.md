# Multi-agent Architecture Search via Agentic Supernet (MaAS)

**CMPE 255 - Data Mining Short Story**  
**Paper:** [arXiv:2502.04180](https://arxiv.org/abs/2502.04180)
## Name: Kalhar Mayurbhai Patel
## SJSU ID: 019140511

---

## 📋 Assignment Overview

-1) Medium Article Link:
-2) PPT Link:
-3) Youtube Video Link: 

This assignment presents a comprehensive analysis of the MaAS (Multi-agent Architecture Search via Agentic Supernet) framework, a groundbreaking approach to automated multi-agent AI system design published at ICML 2025. The work is highly relevant to CMPE 255 as it addresses core data mining concepts including:

- **Automated Machine Learning (AutoML)**: Architecture search and optimization
- **Meta-Learning**: Learning to learn and self-evolution
- **Multi-Agent Systems**: Collaborative AI systems
- **Resource-Aware Computing**: Balancing performance with computational efficiency
- **Transfer Learning**: Cross-dataset and cross-model generalization

---

## 📂 Repository Structure

```
.
├── README.md                          # This file
├── 1274_Multi_agent_Architecture_.pdf # Original research paper
├── slide_deck_with_notes.md          # Presentation slides with speaker notes
```

---

## 📄 Paper Information

**Title:** Multi-agent Architecture Search via Agentic Supernet  
**Authors:** Guibin Zhang, Luyang Niu, Junfeng Fang, Kun Wang, Lei Bai, Xiang Wang  
**Institutions:**  
- National University of Singapore
- Tongji University  
- Nanyang Technological University
- Shanghai AI Laboratory
- University of Science and Technology of China

**Published:** ICML 2025 (Proceedings of the 42nd International Conference on Machine Learning)  
**Code Repository:** https://github.com/bingreeky/MaAS

---

## 🎯 Key Contributions of the Paper

### 1. **Paradigm Shift in Multi-Agent System Design**
- Moves from searching for a single optimal multi-agent architecture to optimizing a **distribution** of architectures
- Introduces the concept of **agentic supernet**: a probabilistic, continuous distribution of multi-agent systems

### 2. **Query-Dependent Architecture Sampling**
- Automatically tailors multi-agent systems to each query's difficulty and domain
- Implements adaptive resource allocation based on task complexity
- Achieves 6-45% of inference costs compared to existing methods

### 3. **Cost-Constrained Self-Evolution**
- Joint optimization of both architecture probabilities and operator internals
- Novel use of **textual gradients** for updating natural language components
- Balances solution quality with computational efficiency

### 4. **Superior Performance and Efficiency**
- 0.54% to 16.89% performance improvements across benchmarks
- 85% reduction in training costs, 75-94% reduction in inference costs
- Strong cross-dataset and cross-LLM-backbone transferability

---

## 🔬 Technical Deep Dive

### Core Concepts

#### Agentic Supernet
A multi-layer probabilistic distribution over possible multi-agent architectures:
- **Layers (L)**: Typically 4 layers in experiments
- **Operators**: CoT, Debate, ReAct, Ensemble, Self-Refine, Early-Exit
- **Distribution (π)**: Conditional probabilities

**Mathematical Formulations:**

1. **Agentic Operator Definition:**
```
O = {{Mᵢ}ᵢ₌₁ᵐ, P, {Tᵢ}ᵢ₌₁ⁿ}
where:
- Mᵢ ∈ M (LLM instances)
- P ∈ P (prompts)  
- Tᵢ ∈ T (tools)
- m = number of LLM agents
- n = number of tools
```

2. **Supernet Joint Distribution:**
```
p(G) = ∏ₗ₌₁ᴸ ∏ₒ∈O πₗ(O)^𝕀[O∈Vₗ]

where:
- πₗ(O) = p(O | A₁:ₗ₋₁) is conditional probability of operator O at layer ℓ
- 𝕀[O∈Vₗ] is indicator function for operator inclusion
- Vₗ is the set of active operators at layer ℓ
- L is the number of layers
```

3. **Optimization Objective:**
```
max P(G|q) E[(q,a)~D, G~P(G|q)] [U(G; q, a) - λ·C(G; q)]

subject to: G ⊂ A

where:
- U(G; q, a) measures solution quality/utility
- C(G; q) measures computational cost (tokens, API calls)
- λ is the trade-off parameter balancing performance and efficiency
- D is the dataset of queries and answers
```

4. **Gradient Estimation (Monte Carlo):**
```
∇π L ≈ (1/K) Σ(q,a)∈D Σₖ₌₁ᴷ [mₖ ∇π p(Gₖ)]

where mₖ = p(a|q, Gₖ)/Σᵢ p(a|q, Gᵢ) - λ·C(Gₖ; q)/Σᵢ C(Gᵢ; q)

- First term promotes architectures that generate correct solutions
- Second term penalizes expensive architectures
- mₖ is cost-aware importance weight
- K is the number of samples (typically K=4)
```

5. **Textual Gradient:**
```
∇O L = T_P ⊕ T_T ⊕ T_N

where:
- T_P: Prompt gradient (natural language suggestions for prompt updates)
- T_T: Temperature gradient (suggestions for temperature adjustments)
- T_N: Node structure gradient (suggestions for operator modifications)
```

### Three-Stage Process

#### Stage 1: Query-Aware Architecture Sampling

The controller network samples architectures conditioned on queries:

```
Q_φ(G|q, π, O) = ∏ₗ₌₁ᴸ [πₗ(Vₗ|q, {Vₕ}ₕ₌₁ˡ⁻¹) · 𝕀[O_exit ∉ Vₗ]]
                  + 𝕀[O_exit ∈ Vₗ] · δ(ℓ - ℓ_exit)

where:
- Vₗ is selected operators at layer ℓ
- O_exit is the early-exit operator
- δ is Kronecker delta function
- ℓ_exit is the exit layer
```

Operator selection uses threshold-based MoE:

```
Vₗ = {Oₗ₁, Oₗ₂, ..., Oₗₜ}
t = arg min_{k∈{1,...,|O|}} Σⱼ<ₖ S↓[j] > threshold

where:
- S = [S₁, ..., S|O|] are activation scores
- S↓ = sort(S, descending)
- Sᵢ = FFN(v(q) || Σ_{O∈V₁} v(O) || ... || Σ_{O∈Vₗ₋₁} v(O))
- v(·) is the embedding function
```

#### Stage 2: Execution

```
p(a|q, π, O) = ∫ e(a|G) Q_φ(G|q, π, O) dG

where:
- e(a|G) is the execution function
- Integration marginalizes over all possible architectures
```

#### Stage 3: Cost-Constrained Evolution

Joint optimization of distribution and operators:

```
Loss = -p(a|q, π, O) + λ·C(G; q)
```

### Experimental Results

#### Benchmarks Evaluated
- **Math Reasoning**: GSM8K, MATH, MultiArith
- **Code Generation**: HumanEval, MBPP
- **Tool Usage**: GAIA

#### Performance Highlights
| Dataset | MaAS | Best Baseline | Improvement |
|---------|------|---------------|-------------|
| GSM8K | 92.30% | 91.16% (AFlow) | +1.14% |
| MATH | 51.82% | 51.28% (AFlow) | +0.54% |
| HumanEval | 92.85% | 90.93% (AFlow) | +1.92% |
| MBPP | 82.17% | 81.67% (AFlow) | +0.50% |
| MultiArith | 98.80% | 97.77% (AgentSquare) | +1.03% |
| GAIA (avg) | 20.69% | 16.61% (TapeAgent) | +4.08% |

#### Efficiency Metrics (MATH Benchmark)
| Method | Training Cost | Training Time | Inference Cost |
|--------|---------------|---------------|----------------|
| MaAS | $3.38 | 53 min | $0.42/query |
| AFlow | $22.50 | 184 min | $1.66/query |
| DyLAN | $13.01 | 508 min | $2.89/query |
| LLM-Debate | - | - | $6.76/query |

**Key Savings:** 
- 85% training cost reduction vs. AFlow
- 75% inference cost reduction vs. AFlow  
- 94% inference cost reduction vs. LLM-Debate

---
## 🔑 Key Takeaways

### Why This Paper Matters for Data Mining

1. **AutoML Innovation**: Extends architecture search from neural networks to multi-agent systems
2. **Meta-Learning**: Self-evolving systems that learn optimal problem decomposition strategies
3. **Resource Optimization**: Critical for deploying data mining systems at scale
4. **Adaptive Systems**: Dynamic resource allocation based on query complexity
5. **Transfer Learning**: Learned patterns generalize across domains and models

### Relevance to CMPE 255 Topics

This work connects to multiple course themes:

- **Automated ML**: Architecture search and hyperparameter optimization
- **Ensemble Methods**: Combining multiple models/agents strategically
- **Meta-Learning**: Learning to learn from task distributions
- **Computational Efficiency**: Resource-aware algorithm design
- **Multi-Agent Coordination**: Distributed problem solving
- **Probabilistic Methods**: Distribution-based optimization
- **Gradient-Based Optimization**: Novel textual gradient approach

### Real-World Impact

**Educational Applications:**
- Adaptive tutoring that scales complexity to student level
- Personalized learning paths based on query difficulty

**Software Development:**
- Code assistants: lightweight for simple tasks, thorough for complex debugging
- Automated testing with resource-aware test generation

**Research & Analysis:**
- Literature review: efficient skimming with deep dives when needed
- Data analysis: simple queries get quick answers, complex ones get comprehensive treatment

**Customer Support:**
- Tier-based routing: simple queries → fast responses, complex → sophisticated reasoning
- Cost-effective scaling while maintaining quality

### Future Research Directions

1. **Larger Operator Libraries**: Scaling to 50+ operators
2. **Online Adaptation**: Continual learning from deployment feedback
3. **Multi-Objective Optimization**: Beyond accuracy/cost (latency, privacy, carbon footprint)
4. **Hierarchical Supernets**: Nested distributions for even more complex reasoning
5. **Cross-Modal Extension**: Vision-language and multimodal agents
6. **Theoretical Analysis**: Convergence guarantees and sample complexity bounds

---

## 📚 References

### Primary Source
Zhang, G., Niu, L., Fang, J., Wang, K., Bai, L., & Wang, X. (2025). Multi-agent Architecture Search via Agentic Supernet. *Proceedings of the 42nd International Conference on Machine Learning* (ICML 2025). PMLR 267.

### Related Work Cited

**Automated Agentic Systems:**
- AFlow: Zhang et al. (2024) - Monte Carlo tree search for workflow automation
- ADAS: Hu et al. (2024) - Heuristic search for agent design
- AgentSquare: Shang et al. (2024) - Modular agent search with Bayesian optimization
- GPTSwarm: Zhuge et al. (2024) - Language agents as optimizable graphs

**Multi-Agent Frameworks:**
- AutoGen: Wu et al. (2023) - Conversation-based multi-agent framework
- MetaGPT: Hong et al. (2023) - Meta programming for agent collaboration
- AgentVerse: Chen et al. (2023) - Multi-agent collaborative framework
- LLM-Debate: Du et al. (2023) - Improving reasoning through multi-agent debate

**Core Techniques:**
- Chain-of-Thought: Wei et al. (2022) - Step-by-step reasoning prompting
- ReAct: Yao et al. (2023) - Synergizing reasoning and acting
- Self-Consistency: Wang et al. (2023) - Aggregating multiple reasoning paths
- DARTS: Liu et al. (2018) - Differentiable architecture search

**Neural Architecture Search:**
- NAS Survey: Ren et al. (2021) - Comprehensive NAS overview
- Supernet Training: White et al. (2023) - Insights from 1000 NAS papers
- SNAS: Xie et al. (2018) - Stochastic neural architecture search

---

## 🛠️ Tools and Technologies

**Development:**
- Python for implementation
- PyTorch for neural components
- LLM APIs: GPT-4o-mini, Qwen-2.5-72b, Llama-3.1-70b

**Evaluation:**
- Benchmarks: HumanEval, MBPP, GSM8K, MATH, MultiArith, GAIA
- Metrics: Pass@1 (code), Accuracy (math/tool use), Token cost, Wall-clock time

**Analysis:**
- Statistical testing for significance
- Ablation studies for component importance
- Sensitivity analysis for hyperparameters

---

## 📊 Additional Insights

### Ablation Study Results

| Component Removed | HumanEval Impact | MATH Impact | Inference Cost |
|-------------------|------------------|-------------|----------------|
| Textual Gradient | -2.68% | -3.59% | No change |
| Early Exit | -1.41% | -0.29% | +40-65% |
| Cost Constraint | +0.09% | -0.63% | +35-49% |

**Interpretation:**
- Textual gradient is critical for performance (enables self-evolution)
- Early exit is critical for efficiency (minimal performance impact)
- Cost constraint balances efficiency without major performance loss

### Sensitivity Analysis

**Number of Layers (L):**
- L=2: 89.5% accuracy, lower cost
- L=4: 92.8% accuracy, optimal balance  
- L=6: 93.1% accuracy, diminishing returns

**Cost Penalty (λ):**
- Higher λ → more cost-efficient, slight performance drop
- Lower λ → higher performance, increased costs
- Tunable based on deployment priorities

**Sampling Count (K):**
- K=2: Suboptimal, high variance
- K=4: Optimal for stable estimates
- K>4: Minimal improvement

---
**Original Paper Repository:** https://github.com/bingreeky/MaAS
---

## 📄 License and Attribution

This assignment is submitted for CMPE 255 - Data Mining course. All analysis and written content is original work. The paper being analyzed is:

```bibtex
@inproceedings{zhang2025maas,
  title={Multi-agent Architecture Search via Agentic Supernet},
  author={Zhang, Guibin and Niu, Luyang and Fang, Junfeng and Wang, Kun and Bai, Lei and Wang, Xiang},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025},
  organization={PMLR}
}
```

---

**Last Updated:** November 2024  
**Course:** CMPE 255 - Data Mining  
**Assignment:** Individual Paper Analysis and Presentation
